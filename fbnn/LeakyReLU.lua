local LeakyReLU, parent = torch.class('fbnn.LeakyReLU', 'nn.PReLU')

function LeakyReLU:__init(p)
  parent.__init(self)
  self.weight:fill(p)
  self.gradWeight:fill(0)
end

function LeakyReLU:__tostring__()
  return torch.type(self) .. string.format('(%g)', self.weight[1])
end

function LeakyReLU:accGradParameters(input, gradOutput, scale)
end

function LeakyReLU:zeroGradParameters()
end

function LeakyReLU:updateParameters(learningRate)
end
