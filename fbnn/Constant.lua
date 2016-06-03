------------------------------------------------------------------------
--[[ Constant ]]--
-- author : Nicolas Leonard
-- Outputs a constant value given an input.
------------------------------------------------------------------------
local Constant, parent = torch.class("fbnn.Constant", "nn.Module")

function Constant:__init(value)
   self.value = value
   if torch.type(self.value) == 'number' then
      self.value = torch.Tensor{self.value}
   end
   assert(torch.isTensor(self.value), "Expecting number or tensor at arg 1")
   parent.__init(self)
end

function Constant:updateOutput(input)
   -- "input:size(1)"" makes the assumption that you're in batch mode
   local vsize = self.value:size():totable()
   self.output:resize(input:size(1), table.unpack(vsize))
   local value = self.value:view(1, table.unpack(vsize))
   self.output:copy(value:expand(self.output:size()))
   return self.output
end

function Constant:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   return self.gradInput
end
