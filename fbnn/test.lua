local ok = pcall(function() require 'fb.luaunit' end)
if not ok then
   print('For running tests, please manually install fb.luaunit from fblualib.')
   print('fblualib does not have rockspecs yet')
   return
end
require 'fbtorch'
require 'nn'

local pl = require'pl.import_into'()

local mytester = torch.Tester()

local fbnntest = {}

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

mytester:add(fbnntest)

function nn.fbnntest(tests)
    math.randomseed(os.time())
    mytester:run(tests)
end
