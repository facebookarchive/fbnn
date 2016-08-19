require 'math'
require 'nn'

local SoftPlusLSEMinusLSECriterion, parent =
    torch.class('nn.SoftPlusLSEMinusLSECriterion', 'nn.Criterion')

-- loss(x) = log(1 + SumExp(x)) = logSumExp(x)
function SoftPlusLSEMinusLSECriterion:__init(sizeAverage)
   parent.__init(self)
   if sizeAverage ~= nil then
       self.sizeAverage = sizeAverage
   else
       self.sizeAverage = true
   end
   --self.beta = beta or 1
   self.threshold = 20 -- avoid overflow
   self.LSE = torch.Tensor()
end

local function softplus(x)
    if x > 20 then
        return x
    else
        return math.log(1 + math.exp(x))
    end
end

function SoftPlusLSEMinusLSECriterion:updateOutput(input)
    local max_val = torch.max(input, 2)
    input = input - max_val:expand(input:size())
    self.LSE = input:exp():sum(2):log()
    self.LSE:add(max_val)
    self.SoftPlusLSE = self.LSE:clone()
    self.SoftPlusLSE:apply(softplus)
    self.output = (self.SoftPlusLSE - self.LSE):sum()
    return self.output
end

function SoftPlusLSEMinusLSECriterion:updateGradInput(input)
    self.gradInput = torch.exp(input - self.SoftPlusLSE:expand(input:size()))
    self.gradInput:add(-torch.exp(input - self.LSE:expand(input:size())))
    return self.gradInput
end
