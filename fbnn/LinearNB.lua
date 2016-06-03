local LinearNB, parent = torch.class('nn.LinearNB', 'nn.Linear')

--[[

This file is still here because of backward compatibility. It is preferred you
use `nn.Linear(input, output, false)` instead.

]]--

function LinearNB:__init(inputSize, outputSize)
   parent.__init(self, inputSize, outputSize, false)
end
