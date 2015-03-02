-- Copyright 2004-present Facebook. All Rights Reserved.

--[[
`TrueNLLCriterion` computes the negative log-loss criterion directly.
]]
local TrueNLLCriterion, parent = torch.class('nn.TrueNLLCriterion',
                                             'nn.Criterion')

-- For numerical stability
local eps = 0.00000001

function TrueNLLCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function TrueNLLCriterion:updateOutput(input, target)
   if input:dim() == 1 then
      self.output = -math.log(input[target] + eps)
   elseif input:dim() == 2 then
      local output = 0
      for i=1,target:size(1) do
         output = output - math.log(input[i][target[i]] + eps)
      end
      if self.sizeAverage then
         output = output / target:size(1)
      end
      self.output = output
   else
      error('matrix or vector expected')
   end
   return self.output
end

function TrueNLLCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()

  if input:dim() == 1 then
      self.gradInput[target] = -1 / (input[target] + eps)
   else
      local z = -1
      if self.sizeAverage then
         z = z / target:size(1)
      end
      local gradInput = self.gradInput
      for i=1,target:size(1) do
         gradInput[i][target[i]] = z / (input[i][target[i]] + eps)
      end
   end

   return self.gradInput
end
