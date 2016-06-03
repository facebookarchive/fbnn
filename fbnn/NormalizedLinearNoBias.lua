local Linear, parent = torch.class('fbnn.NormalizedLinearNoBias', 'nn.Linear')
--[[
    This module creates a Linear layer, but with no bias component.
    In training mode, it constantly self-normalizes it's weights to
    be of unit norm.
]]--

function Linear:__init(inputSize, outputSize)
    parent.__init(self, inputSize, outputSize)
    self.bias:zero()
end

function Linear:updateOutput(input)
    if self.train then
        -- in training mode, renormalize the weights
        -- before every forward call
        self.weight:div(self.weight:norm())
        local scale = math.sqrt(self.weight:size(1))
        self.weight:mul(scale)
    end
    return parent.updateOutput(self, input)
end

function Linear:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    if input:dim() == 1 then
        self.gradWeight:addr(scale, gradOutput, input)
    elseif input:dim() == 2 then
        self.gradWeight:addmm(scale, gradOutput:t(), input)
    end
end
