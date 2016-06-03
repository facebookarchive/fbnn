local LinearNoBackprop, parent = torch.class('nn.LinearNoBackprop', 'nn.Linear')
-- This is like Linear, except that it does not backpropagate gradients w.r.t.
-- input.
function LinearNoBackprop:__init(inputSize, outputSize)
   parent.__init(self, inputSize, outputSize)
end

function LinearNoBackprop:updateGradInput(input, gradOutput)
   if self.gradInput then
      self.gradInput:resizeAs(input)
      self.gradInput:zero()
      return self.gradInput
   end
end
