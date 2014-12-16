
--[[
A faster variant of `nn.SparseLinear` that imposes stricter
preconditions to speed up `updateParameters`.
]]
local SparseLinear, parent = torch.class('fbnn.SparseLinear', 'nn.SparseLinear')

function SparseLinear:__init(inputSize, outputSize, useSparseUpdate)
  parent.__init(self, inputSize, outputSize)
  self.useSparseUpdate = useSparseUpdate
end

function SparseLinear:updateParameters(learningRate)
  if self.useSparseUpdate then
    if not self.lastInput then
      error('lastInput not available. call accGradParameters first')
    end
    self.weight.nn.SparseLinear_updateParameters(self, learningRate)
  else
    parent.updateParameters(self, learningRate)
  end
end

function SparseLinear:zeroGradParameters()
  if self.useSparseUpdate then
    if not self.lastInput then
      error('lastInput not available. call accGradParameters first')
    end
    self.weight.nn.SparseLinear_zeroGradParameters(self)
  else
    parent.zeroGradParameters(self)
  end
end
