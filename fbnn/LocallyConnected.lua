-- Copyright 2004-present Facebook. All Rights Reserved.

--
-- LocallyConnected layer, see
-- https://code.google.com/p/cuda-convnet/wiki/LayerParams#
--         Locally-connected_layer_with_unshared_weights
--
require('torch')

local LocallyConnected, parent = torch.class('nn.LocallyConnected',
                                             'nn.Module')
function LocallyConnected:__init(nInputPlane, iW, iH, nOutputPlane, kW, kH,
                                 dW, dH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   -- validate inputs
   assert(nInputPlane > 0, "Number of input planes must be positive.")
   assert(iW > 0, "Input image width must be positive.")
   assert(iH > 0, "Input image height must be positive.")
   assert(nOutputPlane > 0, "Number of output planes must be positive.")
   assert(0 < kW, "Kernel width must be positive.")
   assert(0 < kH, "Kernel height must be positive.")
   assert(0 < dW, "Column stride must be positive.")
   assert(0 < dH, "Row stride must be positive.")
   assert(kW <= iW, "Kernel width must not exceed input image width.")
   assert(kH <= iH, "Kernel height must not exceed input image height.")

   -- initialize module state
   self.nInputPlane = nInputPlane
   self.iW = iW
   self.iH = iH
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH

   local oW, oH = self:outputSize()
   self.weight     = torch.Tensor(nOutputPlane, oH, oW, nInputPlane, kH, kW)
   self.gradWeight = torch.Tensor():resizeAs(self.weight)
   self.bias       = torch.Tensor(nOutputPlane, oH, oW)
   self.gradBias   = torch.Tensor():resizeAs(self.bias)

   if 'torch.CudaTensor' == torch.getdefaulttensortype() then
      self.weight     = self.toInterleaved(self.weight, true)
      self.gradWeight = self.toInterleaved(self.gradWeight, true)
      self.bias       = self.toInterleaved(self.bias, true)
      self.gradBias   = self.toInterleaved(self.gradBias, true)
   end

   self.input_cache = torch.Tensor() -- cache for CUDA
   self.gradOutput_cache = torch.Tensor() -- cache for CUDA
   self.gradOutputCacheIsValid = false
   self.output = torch.Tensor()
   self.gradInput = torch.Tensor()

   self:reset()
end

function LocallyConnected:outputSize()
   local oW = math.floor((self.iW - self.kW) / self.dW + 1)
   local oH = math.floor((self.iH - self.kH) / self.dH + 1)

   return oW, oH
end

function LocallyConnected:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1 / math.sqrt(self.kW * self.kH * self.nInputPlane)
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
end

function LocallyConnected:updateOutput(input)
   -- validate inputs
   assert(input:dim() == 3 or input:dim() == 4,
          "Invalid input. Must be 3- or 4-D")

   if input:dim() == 3 then
      assert(input:size(1) == self.nInputPlane,
             "Number of input planes mismatch")
      assert(input:size(2) == self.iH, "Input height mismatch")
      assert(input:size(3) == self.iW, "Input width mistmatch")
   else
      assert(input:size(2) == self.nInputPlane,
             "Number of input planes mismatch")
      assert(input:size(3) == self.iH, "Input height mismatch")
      assert(input:size(4) == self.iW, "Input width mismatch")
   end

   -- resize output based on configuration
   -- this can't be done in the constructor because we don't bake
   -- batch size into the layer state. (perf note: tensor resize to same size
   -- is a no-op.)
   local size = input:size()
   local oW, oH = self:outputSize()
   if (input:dim() == 3) then
      size[1] = self.nOutputPlane
      size[2] = oH
      size[3] = oW
   else
      size[2] = self.nOutputPlane
      size[3] = oH
      size[4] = oW
   end

   self.gradOutputCacheIsValid = false -- invalidate gradOutput cache

   self.output = self.output:resize(size)

   local result = input.nn.LocallyConnected_updateOutput(self, input)

   return result
end

function LocallyConnected:updateGradInput(input, gradOutput)
   -- Invocation of this method sets the gradOutput cache, i.e. makes the
   -- cache valid.
   local result =  input.nn.LocallyConnected_updateGradInput(self, input,
                                                             gradOutput)
   self.gradOutputCacheIsValid = true

   return result
end

function LocallyConnected:accGradParameters(input, gradOutput, scale)
   scale = scale or 1.0
   input.nn.LocallyConnected_accGradParameters(self, input, gradOutput, scale)
   -- this method leaves gradOutputCache in valid state.
   self.gradOutputCacheIsValid = true
end

-- Change a 3-d or 4-d tensor from standard, planar Torch layout (P x H x W) or
-- (B x P x H x W) to interleaved layout (H x W x P) or (B x H x W x P).
-- Change a 6-d weight tensor from planar (P_o x H_o x W_o x P_i x H_k x W_k) to
-- interleaved format (H_o x W_o x H_k x W_k x P_o x P_i). The make_contiguous
-- flag controls if the result tensor is guaranteed to be contiguous.
function LocallyConnected.toInterleaved(tensor, make_contiguous)
   if tensor:dim() == 4 then          -- B x P x H x W
      tensor = tensor:transpose(2, 4) -- B x W x H x P
      tensor = tensor:transpose(2, 3) -- B x H x W x P
   elseif tensor:dim() == 3 then      -- P x H x W
      tensor = tensor:transpose(1, 3) -- W x H x P
      tensor = tensor:transpose(1, 2) -- H x W x P
   elseif tensor:dim() == 6 then      -- P_o x H_o x W_o x P_i x H_k x W_k
      tensor = tensor:transpose(1, 3) -- W_o x H_o x P_o x P_i x H_k x W_k
      tensor = tensor:transpose(1, 2) -- H_o x W_o x P_o x P_i x H_k x W_k
      tensor = tensor:transpose(3, 5) -- H_o x W_o x H_k x P_i x P_o x W_k
      tensor = tensor:transpose(4, 6) -- H_o x W_o x H_k x W_k x P_o x P_i
   else
      error('Unsupported tensor size')
   end
   if make_contiguous then
      tensor = tensor:contiguous()
   end

   return tensor
end


-- Inverse operation of toInterleaved.
function LocallyConnected.toPlanar(tensor, make_contiguous)
   if tensor:dim() == 4 then          -- B x H x W x P
      tensor = tensor:transpose(2, 3) -- B x W x H x P
      tensor = tensor:transpose(2, 4) -- B x P x H x W
   elseif tensor:dim() == 3 then      -- H x W x P
      tensor = tensor:transpose(1, 2) -- W x H x P
      tensor = tensor:transpose(1, 3) -- P x H x W
   elseif tensor:dim() == 6 then      -- H_o x W_o x H_k x W_k x P_o x P_i
      tensor = tensor:transpose(4, 6) -- H_o x W_o x H_k x P_i x P_o x W_k
      tensor = tensor:transpose(3, 5) -- H_o x W_o x P_o x P_i x H_k x W_k
      tensor = tensor:transpose(1, 2) -- W_o x H_o x P_o x P_i x H_k x W_k
      tensor = tensor:transpose(1, 3) -- P_o x H_o x W_o x P_i x H_k x W_k
   else
      error('Unsupported tensor size')
   end
   if make_contiguous then
      tensor = tensor:contiguous()
   end

   return tensor
end

-- Type conversion.
-- The trick here is to convert the various tensors to and from
-- cuda layout when a conversion to or from host to GPU takes place.
function LocallyConnected:type(type)
   -- requesting type
   if not type then
      return nn.Module.type(self)
   end

   -- converting to or from a CUDA tensor data-layout must be adjusted...
   if self:type() == 'torch.CudaTensor' and type ~= 'torch.CudaTensor' then
      -- if going from CUDA to host type convert toPlanar(...)
      self.weight = self.toPlanar(self.weight, true)
      self.gradWeight = self.toPlanar(self.gradWeight, true)
      self.bias = self.toPlanar(self.bias, true)
      self.gradBias = self.toPlanar(self.gradBias, true)
   elseif self:type() ~= 'torch.CudaTensor' and type == 'torch.CudaTensor' then
      -- if going from host type to CUDA convert toInterleaved(...)
      self.weight = self.toInterleaved(self.weight, true)
      self.gradWeight = self.toInterleaved(self.gradWeight, true)
      self.bias = self.toInterleaved(self.bias, true)
      self.gradBias = self.toInterleaved(self.gradBias, true)
   end

   return nn.Module.type(self, type)
end
