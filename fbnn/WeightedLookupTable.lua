-- Copyright 2004-present Facebook. All Rights Reserved.

require('nn')

local WeightedLookupTable, parent =
   torch.class('nn.WeightedLookupTable', 'nn.LookupTable')

WeightedLookupTable.__version = 2

function WeightedLookupTable:__init(nIndex, nOutput)
   parent.__init(self, nIndex, nOutput)
   self._gradOutput = torch.Tensor()
   self._embeddings = torch.Tensor()
end

--[[
Parameters:
* `Input` should be an n x 2 tensor where the first column is dictionary indexes
   and the second column is weights.
]]
function WeightedLookupTable:updateOutput(input)
   if input:dim() ~= 2 or input:size(2) ~= 2 then
      error('`Input` should be an n x 2 tensor')
   end

   local indices = input:select(2, 1)
   local weights = input:select(2, 2)

   self._embeddings = parent.updateOutput(self, indices)

   -- Multiply each row of output by the input weight
   input.nn.WeightedLookupTable_scaleByWeight(self.output, self._embeddings,
                                              weights)
   return self.output
end

function WeightedLookupTable:accGradParameters(input, gradOutput, scale)
   local indices = input:select(2, 1)
   local weights = input:select(2, 2)

   self._gradOutput = self._gradOutput or torch.Tensor()
   self._gradOutput:resizeAs(gradOutput)

   -- Multiply each row of gradOutput by input weight
   input.nn.WeightedLookupTable_scaleByWeight(self._gradOutput, gradOutput, weights)

   parent.accGradParameters(self, indices, self._gradOutput, scale)
end

function WeightedLookupTable:sharedAccUpdateGradParameters(input, gradOutput, lr)
   -- we do not need to accumulate parameters when sharing:
   self:defaultAccUpdateGradParameters(input, gradOutput, lr)
end
