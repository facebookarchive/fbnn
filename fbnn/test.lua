local ok = pcall(function() require 'fb.luaunit' end)
if not ok then
   print('For running tests, please manually install fb.luaunit from fblualib.')
   print('fblualib does not have rockspecs yet')
   return
end
require 'nn'

local precision = 1e-5

local pl = require'pl.import_into'()

local mytester = torch.Tester()
local jac = nn.Jacobian

local fbnntest = {}

local function assertTensorEq(a, b, epsilon)
    local epsilon = epsilon or 0.000001
    local diff = a - b
    assert(diff:abs():max() < epsilon)
end

function fbnntest.Optim_weight_bias_parameters()
    local n = nn.Sequential()
    n:add(nn.Linear(10, 10))
    n:add(nn.Tanh())
    n:add(nn.Add(10))

    for i = 1, 3 do
        local cur_mod = n:get(1)
        local params = nn.Optim.weight_bias_parameters(cur_mod)
        local has_bias = cur_mod.bias ~= nil
        local has_weight = cur_mod.weight ~= nil
        if not has_bias and not has_weight then
            mytester:asserteq(pl.tablex.size(params), 0)
        else
            mytester:asserteq(pl.tablex.size(params), 2)
            if has_weight then
                mytester:assert(not params[1].is_bias)
                mytester:asserteq(params[1][1], cur_mod.weight)
                mytester:asserteq(params[1][2], cur_mod.gradWeight)
            else
                mytester:assert(not params[1])
            end
            if has_bias then
                mytester:assert(params[2].is_bias)
                mytester:assert(params[2][1])
                mytester:assert(params[2][2])
            else
                mytester:assert(not params[2])
            end
        end
    end
end

function fbnntest.CachingLookupTableCoherence()
    -- Make sure that we don't lose writes even with multiple caches
    -- attached
    local function buildCaches(numCaches, rows, cacheRows, cols)
        local lut = nn.LookupTable(rows, cols)
        -- Nice, even values are easier to debug.
        lut.weight = torch.range(1, rows * cols):reshape(rows, cols)
        local caches = { }
        for i = 1, numCaches do
            table.insert(caches, nn.CachingLookupTable(lut, cacheRows, cols))
        end
        return lut, caches
    end
    local cases = {
        -- rows, cols, cacheRows, numCaches, numUpdates
        {  1,     1,    1,         1,         100, },
        {  100,   10,   100,       2,         200, },
        {  500,   100,  100,       2,         500 },
        {  500,   100,  500,       2,         2000 },
    }
    for _,case in pairs(cases) do
        print(case)
        local rows, cols, cacheRows, numCaches, numUpdates = unpack(case)
        local lut, caches = buildCaches(numCaches, rows, cacheRows, cols)
        local lutClone = lut:clone()

        for j = 1,numUpdates do
            local start = math.random(rows)
            local finish = math.min(start + math.random(100), rows)
            local rows = torch.range(start, finish)
            local n = rows:size(1)
            local grad = torch.randn(n, cols)
            for i =1,rows:size(1) do
               lutClone.weight[rows[i]]:add(grad[i] * -#caches)
            end
            for _,cache in ipairs(caches) do
                assert(cache.accUpdateGradParameters ==
                       nn.CachingLookupTable.accUpdateGradParameters)
                cache:accUpdateGradParameters(rows, grad, 1.0)
            end
        end
        for _,cache in ipairs(caches) do
            cache:flush()
        end
        assertTensorEq(lutClone.weight, lut.weight)
    end
end

function fbnntest.testLoGLayer()

    -- load image:
    require 'image'
    local im = image.lena()

    -- test forward pass in simple net:
    local net = nn.Sequential()
    local sigma = 1
    net:add(nn.LaplacianOfGaussian(3, sigma))
    local filteredIm = net:forward(im)
    assert(filteredIm)
    assert(im:size(1) == filteredIm:size(1))
    assert(im:size(2) == filteredIm:size(2))
    assert(im:size(3) == filteredIm:size(3))
end

local function criterionJacobianTest(cri, input, target)
   local eps = 1e-6
   local _ = cri:forward(input, target)
   local dfdx = cri:backward(input, target)
   -- for each input perturbation, do central difference
   local centraldiff_dfdx = torch.Tensor():resizeAs(dfdx)
   local input_s = input:storage()
   local centraldiff_dfdx_s = centraldiff_dfdx:storage()
   for i=1,input:nElement() do
      -- f(xi + h)
      input_s[i] = input_s[i] + eps
      local fx1 = cri:forward(input, target)
      -- f(xi - h)
      input_s[i] = input_s[i] - 2*eps
      local fx2 = cri:forward(input, target)
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx_s[i] = cdfx
      -- reset input[i]
      input_s[i] = input_s[i] + eps
   end

   -- compare centraldiff_dfdx with :backward()
   local err = (centraldiff_dfdx - dfdx):abs():max()
   print(err)
   mytester:assertlt(err, precision,
   'error in difference between central difference and :backward')
end

function fbnntest.testSoftPlusLSEMinusLSECriterion()
   local input = torch.rand(10, 100)
   local cri = nn.SoftPlusLSEMinusLSECriterion()
   criterionJacobianTest(cri, input)
end


function fbnntest.testSoftPlusLSECriterion()
   local input = torch.rand(10, 100)
   local cri = nn.SoftPlusLSECriterion()
   criterionJacobianTest(cri, input)
end

function fbnntest.testLoGNetwork()

    -- load image:
    require 'image'
    local im = image.lena()
    local target = torch.DoubleTensor(1)
    target[1] = 1

    -- set up prediction network:
    local net = nn.Sequential()
    local sigma = 1
    local layer = nn.LaplacianOfGaussian(3, sigma)
    net:add(layer)
    net:add(nn.View(im:nElement()))
    net:add(nn.Linear(im:nElement(), 1))
    net:add(nn.LogSigmoid())
    local criterion = nn.BCECriterion()

    -- forward backward pass:
    local loss = criterion:forward(net:forward(im), target)
    net:backward(im, criterion:backward(net.output, target))
    assert(layer.gradInput:size(1) == im:size(1))
    assert(layer.gradInput:size(2) == im:size(2))
    assert(layer.gradInput:size(3) == im:size(3))
    assert(loss)
end

function fbnntest.testConstantLayer()
    local net = nn.Sequential()
    local cst = math.random()
    net:add(nn.Linear(2,3))
    net:add(fbnn.Constant(cst))
    local output = net:forward(torch.randn(2))
    for i=1,output:size()[1] do
        assert(output[i][1] == cst)
    end
end


function fbnntest.testDoG()

    -- load image:
    require 'image'
    local input = image.scale(image.lena(), 16, 16, 'bilinear')
    local numChannels = input:size(1)

    -- construct module:
    local nOctaves = 3
    local nScalesPerOctave = 4
    local module = nn.DifferenceOfGaussian(
        numChannels,
        nOctaves,
        nScalesPerOctave
    )

    -- test forward pass:
    local output = module:forward(input)
    assert(type(output) == 'table')
    assert(#output == nOctaves)
    for n = 1,nOctaves do
        assert(output[n]:size(1) == nScalesPerOctave * numChannels)
    end

    -- repeat the forward tests in batch mode:
    local batchSize = 8
    local batchInput = input.new(
        batchSize,
        input:size(1),
        input:size(2),
        input:size(3)
    )
    for n = 1,batchSize do
        batchInput[n]:copy(input):add(torch.randn(input:size()), 0.05)
    end
    output = module:forward(batchInput)
    assert(type(output) == 'table')
    assert(#output == nOctaves)
    for n = 1,nOctaves do
        assert(output[n]:size(1) == batchSize)
        assert(output[n]:size(2) == nScalesPerOctave * numChannels)
    end
end

function fbnntest.testUpsample()
    require 'image'
    local factors = torch.DoubleTensor({1, 2, 3, 4})
    local im = image.scale(image.lena(), 32, 32, 'bilinear')

    -- test for single image:
    for n = 1,factors:nElement() do
        local net = nn.Sequential()
        net:add(nn.Upsample(factors[n]))
        local upsampledIm = net:forward(im)
        assert(upsampledIm:size(1) == im:size(1))
        assert(upsampledIm:size(2) == im:size(2) * factors[n])
        assert(upsampledIm:size(3) == im:size(3) * factors[n])
        local recon = net:backward(im, upsampledIm):div(factors[n] * factors[n])
        assert(recon:size(1) == im:size(1))
        assert(recon:size(2) == im:size(2))
        assert(recon:size(3) == im:size(3))
        assert(recon:add(-im):abs():sum() < 1e-5)
    end

    -- test for image batch:
    im:resize(1, im:size(1), im:size(2), im:size(3))
    local batch = im:expand(8, im:size(2), im:size(3), im:size(4))
    batch:add(torch.randn(batch:size()))
    for n = 1,factors:nElement() do
        local net = nn.Sequential()
        net:add(nn.Upsample(factors[n]))
        local upsampledBatch = net:forward(batch)
        assert(upsampledBatch:size(1) == batch:size(1))
        assert(upsampledBatch:size(2) == batch:size(2))
        assert(upsampledBatch:size(3) == batch:size(3) * factors[n])
        assert(upsampledBatch:size(4) == batch:size(4) * factors[n])
        local recon = net:backward(batch, upsampledBatch)
        recon:div(factors[n] * factors[n])
        assert(recon:size(1) == batch:size(1))
        assert(recon:size(2) == batch:size(2))
        assert(recon:size(3) == batch:size(3))
        assert(recon:size(4) == batch:size(4))
        assert(recon:add(-batch):abs():sum() < 1e-5)
    end
end

function fbnntest.WeightedLookupTable()
    local totalIndex = math.random(6,9)
    local nIndex = math.random(3,5)
    local entry_size = math.random(2,5)
    local indices = torch.randperm(totalIndex):narrow(1,1,nIndex)
    local weights = torch.randn(nIndex)
    local module = nn.WeightedLookupTable(totalIndex, entry_size)
    local minval = 1
    local maxval = totalIndex

    local input = torch.Tensor(indices:size(1), 2)
    input:select(2, 1):copy(indices)
    input:select(2, 2):copy(weights)

    local output = module:forward(input)
    for r=1,nIndex do
        for c=1,entry_size do
            mytester:assertlt(math.abs(output[r][c] - module.weight[input[r][1]][c] * input[r][2]), 1e-3, 'incorrect output')
        end
    end
    module:backwardUpdate(input, output, 0.1)
    input:zero()

    -- 1D
    local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight, minval, maxval)
    mytester:assertlt(err, 1e-4, '1D error on weight ')

    local err = jac.testJacobianUpdateParameters(module, input, module.weight, minval, maxval)
    mytester:assertlt(err, 1e-4, '1D error on weight [direct update] ')

    module.gradWeight:zero()
    for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
       mytester:assertlt(err, 1e-4, string.format(
                          '1D error on weight [%s]', t))
    end
end

function fbnntest.IndividualDropout()
    local N = 1024
    local D = 5
    local net = nn.Sequential()
    net:add(fbnn.IndividualDropout(torch.range(0, .8, .8 / (D - 1))))
    local output = net:forward(torch.ones(N, D))
    assert(output)
    assert(output:size(1) == N)
    assert(output:size(2) == D)
end

function fbnntest.FasterLookup()
   local runtest = function(type)
      local totalIndex = math.random(6,9)
      local nIndex = math.random(3,5)
      local entry_size = math.random(2,5)
      local input = torch.randperm(totalIndex):narrow(1,1,nIndex):int()
      local module = nn.FasterLookup(totalIndex, entry_size)
      local minval = 1
      local maxval = totalIndex

      module:type(type)

      local output = module:forward(input)
      module:backwardUpdate(input, output, 0.1)
      input:zero()
      -- 1D
      local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight, minval, maxval)
      mytester:assertlt(err,1e-5, '1D error on weight ')

      local err = jac.testJacobianUpdateParameters(module, input, module.weight, minval, maxval)
      mytester:assertlt(err,1e-5, '1D error on weight [direct update] ')

      module.gradWeight:zero()
      for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
         mytester:assertlt(err, 1e-5, string.format(
                            '1D error on weight [%s]', t))
      end

      -- 2D
      local nframe = math.random(2,5)
      local input = torch.IntTensor(nframe, nIndex):zero()

      local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight, minval, maxval)
      mytester:assertlt(err,1e-5, '2D error on weight ')

      local err = jac.testJacobianUpdateParameters(module, input, module.weight, minval, maxval)
      mytester:assertlt(err,1e-5, '2D error on weight [direct update] ')

      module.gradWeight:zero()
      for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
         mytester:assertlt(err, 1e-5, string.format(
                            '2D error on weight [%s]', t))
      end

      -- IO
      local ferr,berr = jac.testIO(module,input,minval,maxval)
      mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
      mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
   end
   runtest(torch.DoubleTensor():type())
end

mytester:add(fbnntest)

function nn.fbnntest(tests)
    math.randomseed(os.time())
    mytester:run(tests)
end
