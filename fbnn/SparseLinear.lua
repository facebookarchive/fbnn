
--[[
A faster variant of `nn.SparseLinear` that imposes stricter
preconditions to speed up `updateParameters`.
]]
local SparseLinear, parent = torch.class('fbnn.SparseLinear', 'nn.SparseLinear')

function SparseLinear:__init(inputSize,
                             outputSize,
                             useSparseUpdate,
                             skipUpdateGradInput)
  parent.__init(self, inputSize, outputSize)
  self.useSparseUpdate = useSparseUpdate
  self.numBackward = 0

  -- should be true if this is the first layer
  if skipUpdateGradInput then
    self.gradInput = nil
  end
end

function SparseLinear:reshapeKvInput(input)
  if input[1]:dim() == 1 then
    return {input[1]:view(1, -1), input[2]:view(1, -1)}
  else
    return input
  end
end

function SparseLinear:updateOutput(input)
  if type(input) ~= 'table' then
    return parent.updateOutput(self, input)
  else
    input = self:reshapeKvInput(input)
    return self.weight.nn.SparseLinear_updateOutput2(self, input[1], input[2])
  end
end

function SparseLinear:accGradParameters(input, gradOutput, scale)
  if self.useSparseUpdate then
    assert(self.numBackward == 0,
           'you can only call one backward() when using sparse update')
    self.numBackward = self.numBackward + 1
  end

  if type(input) ~= 'table' then
    return parent.accGradParameters(self, input, gradOutput, scale)
  else
    input = self:reshapeKvInput(input)

    if not self.lastInputKey then
      self.lastInputKey = input[1]:clone()
      self.lastInputVal = input[2]:clone()
    else
      self.lastInputKey:resizeAs(input[1]):copy(input[1])
      self.lastInputVal:resizeAs(input[2]):copy(input[2])
    end

    return self.weight.nn.SparseLinear_accGradParameters2(
      self, input[1], input[2], gradOutput, scale)
  end
end

function SparseLinear:updateParameters(learningRate)
  if self.useSparseUpdate then
    assert(self.numBackward == 1, 'must call backward() once')
    if self.lastInputKey then
      self.weight.nn.SparseLinear_updateParameters2(self, learningRate)
    else
      parent.updateParameters(self, learningRate)
    end
  else
    parent.updateParameters(self, learningRate)
  end
end

function SparseLinear:zeroGradParameters()
  if self.useSparseUpdate then
    if self.lastInputKey == nil and self.lastInput == nil then
      assert(self.numBackward == 0, 'oops')
      io.stderr:write('SparseLinear: using full zeroGrad\n')
      parent.zeroGradParameters(self)
    else
      assert(self.numBackward == 1, 'must call backward() once')
      if self.lastInputKey then
        self.weight.nn.SparseLinear_zeroGradParameters2(self)
      else
        parent.zeroGradParameters(self)
      end
    end
    self.numBackward = 0
  else
    parent.zeroGradParameters(self)
  end
end

function SparseLinear:updateGradInput(input, gradOutput)
  if self.gradInput then
    if type(input) ~= 'table' then
      return parent.updateGradInput(self,input, gradOutput)
    else
      error('not supported')
    end
  end
end
