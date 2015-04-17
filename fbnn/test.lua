local ok = pcall(function() require 'fb.luaunit' end)
if not ok then
   print('For running tests, please manually install fb.luaunit from fblualib.')
   print('fblualib does not have rockspecs yet')
   return
end
require 'nn'

local pl = require'pl.import_into'()

local mytester = torch.Tester()

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
                mytester:assert(params[2][1], cur_mod.bias)
                mytester:assert(params[2][2], cur_mod.gradBias)
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
            lutClone:updateRows(rows, grad * -#caches)
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

mytester:add(fbnntest)

function nn.fbnntest(tests)
    math.randomseed(os.time())
    mytester:run(tests)
end
