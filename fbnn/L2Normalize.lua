local No, parent = torch.class('fbnn.L2Normalize', 'nn.Module')

--[[
    This module normalizes it's input to unit euclidean norm
    Authors: Mark Tygert, Soumith Chintala
]]--

function No:__init()
    parent.__init(self)
end


function No:updateOutput(input)
    -- store the number of dimensions of the input
    local ldim = input:nDimension()
    assert(ldim <= 2,
           'This module should only (realistically) '
           .. 'be used for 1-D or 2-D inputs')

    self.output:resizeAs(input)
    self.output:copy(input)

    -- compute the Euclidean norm over the last dimension of the input
    self.norms = self.norms or input.new()
    torch.norm(self.norms, input, 2, ldim)
    -- divide the input by the Euclidean norms to produce the output
    self.output:cdiv(self.norms:expand(self.output:size()))

    return self.output

end


function No:updateGradInput(input,gradOutput)

    self.gradInput:resizeAs(gradOutput)
    self.gradInput:copy(gradOutput)

    -- store the number of dimensions of the input
    local ldim = input:nDimension()

    local proj = self.gradInput

    -- compute the negative of the dot product between the normalized input,
    -- that is, self.output, and gradInput=gradOutput
    local dotprod = proj:clone():cmul(self.output):sum(ldim):mul(-1)
    -- orthogonalize gradInput=gradOutput to the normalized input,
    -- that is, self.output
    proj:add(self.output:clone():cmul(dotprod:expand(proj:size())))
    -- normalize by the norms of the input
    proj:cdiv(self.norms:expand(proj:size()))

    return proj

end
