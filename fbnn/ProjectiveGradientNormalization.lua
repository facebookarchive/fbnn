--[[
   This file implements a projective gradient normalization proposed by Mark Tygert.
   This alters the network from doing true back-propagation.

   The operation implemented is:
   forward:
              Y = X
   backward:
              dL     dL        X      {     X          dL   }
              --  =  --   -  ----  *  |   ----    (.)  --   |
              dX     dY      ||X||    {   ||X||        dY   }
                                  2            2

   where (.) = dot product

   Usage:
   fbnn.ProjectiveGradientNormalization([eps = 1e-5]) -- eps is optional defaulting to 1e-5

   eps is a small value added to the ||X|| to avoid divide by zero
       Defaults to 1e-5

]]--
local BN,parent = torch.class('fbnn.ProjectiveGradientNormalization', 'nn.Module')

function BN:__init(eps)
   parent.__init(self)
   self.eps = eps or 1e-5
end

function BN:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   return self.output
end

function BN:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput)
   local x_norm = input:norm(2) -- L2 norm of x
   if x_norm == 0 then
      self.gradInput:copy(gradOutput)
   else
      local inv_x_norm = 1 / x_norm
      self.gradInput:copy(input):mul(inv_x_norm)
      local proj_norm = torch.dot(self.gradInput:view(-1), gradOutput:view(-1))
      self.gradInput:mul(-proj_norm):add(gradOutput)
   end
   return self.gradInput
end
