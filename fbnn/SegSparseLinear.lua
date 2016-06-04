local SegSparseLinear = torch.class('fbnn.SegSparseLinear', 'nn.Module')

local function getGetter(t)
  if t:numel() == 0 then return nil end

  local p = t:data()
  assert(t:dim() == 1)
  local s1 = t:stride(1)
  local function getter(i1)
    return p[(i1 - 1) * s1]
  end
  return getter
end

local function axpy(n, a, x, y)
  for i = 0, (n - 1) do
    y[i] = y[i] + a * x[i]
  end
end

function SegSparseLinear:__init(inputSize, outputSize, useSparseGrad,
  learningRateMul)
  self.weight = torch.zeros(outputSize, inputSize) -- will be transposed
  self.bias = torch.zeros(outputSize)
  self.output = torch.zeros(outputSize)
  -- sparse gradient of weight
  self.sparseGradWeightPtr = {}
  self._sparseBuf = torch.zeros(outputSize)
  -- using sparse gradient will be a bit slower (D3376618)
  self.useSparseGrad = useSparseGrad or false
  if not self.useSparseGrad then
    self.gradWeight = torch.zeros(inputSize, outputSize)
  end

  self.gradBias = torch.zeros(outputSize)
  self.ones = torch.ones(100000)

  self.learningRateMul = learningRateMul or 1

  self:reset()
end

function SegSparseLinear:reset(stdv)
  if stdv then
    stdv = stdv * math.sqrt(3)
  else
    stdv = 1./math.sqrt(self.weight:size(2))
  end

  self.weight:uniform(-stdv, stdv)
  self.weight = self.weight:t():contiguous()
  self.bias:uniform(-stdv, stdv):mul(1e-6)
end

function SegSparseLinear:setLearningRateMul(learningRateMul)
  self.learningRateMul = learningRateMul
end

function SegSparseLinear:setUseSparseGrad(useSparseGrad)
  if self.useSparseGrad == useSparseGrad then
    return
  end

  if useSparseGrad then
    self.gradWeight = nil
  else
    self.gradWeight = self.weight:clone():zero()
  end
  self.useSparseGrad = useSparseGrad
end

function SegSparseLinear:updateOutput(input)
  local segs, keys, vals = unpack(input)

  assert(self.output:isContiguous() and self.weight:isContiguous())

  local n = segs:numel()
  local m = self.weight:size(2)
  local k = self.weight:size(1)

  local batch_size = input.batch_size or segs:max()

  self.output:resize(batch_size, self.bias:numel()):zero()
  self.output:addr(self.ones:sub(1, batch_size), self.bias)

  assert(n == 0 or (segs:max() <= batch_size and segs:min() >= 1 and
         keys:max() <= k and keys:min() >= 1))

  local segGet = getGetter(segs)
  local valGet  = getGetter(vals)
  local keyGet = getGetter(keys)

  local outputPtr = self.output:data()
  local weightPtr = self.weight:data()

  for i = 1, n do
    axpy(m, valGet(i),
      weightPtr + (keyGet(i) - 1) * m,
      outputPtr + (segGet(i) - 1) * m)
  end

  return self.output
end

function SegSparseLinear:accGradParameters(input, gradOutput, scale)
  local segs, keys, vals = unpack(input)

  assert(self.gradBias:isContiguous() and gradOutput:isContiguous())
  if not self.useSparseGrad then
    assert(self.gradWeight:isContiguous())
  end

  local n = segs:numel()
  local m = self.weight:size(2)
  local k = self.weight:size(1)

  local batch_size = input.batch_size or segs:max()
  assert(gradOutput:size(1) == batch_size, 'inconsistent')

  assert(n == 0 or (segs:max() <= batch_size and segs:min() >= 1 and
         keys:max() <= k and keys:min() >= 1))

  local segGet = getGetter(segs)
  local valGet  = getGetter(vals)
  local keyGet = getGetter(keys)

  if self.useSparseGrad then
    local keySet = {}
    local cnt = 0
    for i = 1, n do
      local key = tonumber(keyGet(i))
      if keySet[key] == nil then
        keySet[key] = cnt
        cnt = cnt + 1
      end
    end
    -- the content of a Tensor after resizing is undetermined
    -- the elements of the resized tensor are contiguous in memory
    self._sparseBuf:resize(cnt, m):zero()
    local sparseBufPtr = self._sparseBuf:data()

    for k, v in pairs(keySet) do
      self.sparseGradWeightPtr[k] = sparseBufPtr + v * m
    end
  end

  local gradOutputPtr = gradOutput:data()
  local gradBiasPtr = self.gradBias:data()

  local gradWeightPtr = self.gradWeight and self.gradWeight:data()

  self.lastInput = input

  for i = 1, n do
    local key = tonumber(keyGet(i))

    local gradWeightRowPtr
    if self.useSparseGrad then
      gradWeightRowPtr = self.sparseGradWeightPtr[key]
    else
      gradWeightRowPtr = gradWeightPtr + (key - 1) * m
    end

    axpy(m, scale * valGet(i),
      gradOutputPtr + (segGet(i) - 1) * m,
      gradWeightRowPtr)
  end

  for i = 1, batch_size do
    axpy(m, scale, gradOutputPtr + (i - 1) * m, gradBiasPtr)
  end
end

function SegSparseLinear:updateParameters(learningRate)
  learningRate = learningRate * self.learningRateMul

  assert(self.lastInput, 'call backward first')
  local keys = self.lastInput[2]

  assert(self.weight:isContiguous() and self.gradBias:isContiguous())

  local n = keys:numel()
  local m = self.weight:size(2)
  local k = self.weight:size(1)

  assert(n == 0 or (keys:max() <= k and keys:min() >= 1))

  local keyGet = getGetter(keys)

  local weightPtr = self.weight:data()
  local gradWeightPtr = self.gradWeight and self.gradWeight:data()

  local updatedKeys = {}

  for i = 1, n do
    local key = tonumber(keyGet(i))
    if not updatedKeys[key] then
      local gradWeightRowPtr
      if self.useSparseGrad then
        gradWeightRowPtr = self.sparseGradWeightPtr[key]
      else
        gradWeightRowPtr = gradWeightPtr + (key - 1) * m
      end

      axpy(m, -learningRate,
        gradWeightRowPtr,
        weightPtr + (key - 1) * m)

      -- zero out dense grad here; zero out sparse grad when resizing
      if not self.useSparseGrad then
        for j = 0, (m - 1) do
          gradWeightRowPtr[j] = 0
        end
      end
      updatedKeys[key] = true
    end

  end
  self.bias:add(-learningRate, self.gradBias)

  -- zero out gradBias here
  self.gradBias:zero()

  self.lastInput = nil
end

function SegSparseLinear:zeroGradParameters()
  -- done in updateParameters. not needed here.
  self.sparseGradWeightPtr = {}
end

function SegSparseLinear:updateGradInput(input, gradOutput)
  -- always assume this is the first layer and we don't back-prop to data
end

function SegSparseLinear:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(1), self.weight:size(2)) ..
      (self.bias == nil and ' without bias' or '')
end
