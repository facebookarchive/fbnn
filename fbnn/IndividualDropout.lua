local async_rng = require 'fb.torch.async_rng'

-- Hack: store RNG externally so that we don't try to serialize it...
local rngs = {}

-- Weak keys, so we don't leak RNG objects if the corresponding
-- IndividualDropout objects are destroyed
setmetatable(rngs, {__mode = 'k'})

local IndividualDropout, Parent =
   torch.class('fbnn.IndividualDropout', 'nn.Module')

--[[

This module implements a dropout layer with dropout level p. The level p can
either b a number (same dropout level for all units), or a Tensor with the same
width as the input (separate dropout level for every unit).

Parameter:
- `p`: the dropout probabilities (the probability that a given activation will
   be dropped) in a Tensor. The number of elements in the Tensor must equal the
   number of variables in a single input. Both batch mode and single-instance
   mode are supported.

]]--

function IndividualDropout:__init(p)
   Parent.__init(self)
   if torch.lt(p, 0):sum() > 0 or torch.ge(p, 1):sum() > 0 then
      error('<IndividualDropout> illegal percentage, must be 0 <= p < 1')
   end
   self:setp(p)
   self.train = true
end

function IndividualDropout:updateOutput(input)

   -- copy input and make buffers correct size:
   assert(input:nDimension() == 2)
   assert(input:size(2) == self.p:nElement())
   self.output   = self.output   or input.new()
   self.noise    = self.noise    or torch.FloatTensor()
   self.noisegpu = self.noisegpu or input.new()
   self.output:resizeAs(input):copy(input)

   -- only apply dropout at training time:
   if self.train then

      -- initialize async samplers if they do not exist yet:
      local rng = rngs[self]
      if not rng then
         rng = {}
         for d = 1,self.p:nElement() do
            rng[d] = async_rng.bernoulli(
               'torch.FloatTensor', 10 * input:size(1), self.p[d]
            )
         end
         rngs[self] = rng
      end

      -- perform sampling and copy to GPU:
      self.noise:resize(input:size())
      for d = 1,self.p:nElement() do
         self.noise:narrow(2, d, 1):copy(rng[d]:generate(input:size(1))[1])
      end
      self.noisegpu:resize(self.noise:size()):copy(self.noise)

      -- normalize so that we don't need to do anything at test time:
      if not self.pgpu then
         self.pgpu = input.new(1, self.p:nElement()):copy(self.p)
      end
      self.noisegpu:cdiv(self.pgpu:expandAs(self.noisegpu))

      -- apply the dropout:
      self.output:cmul(self.noisegpu)
   end
   return self.output
end

function IndividualDropout:updateGradInput(input, gradOutput)
   if self.train then
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      self.gradInput:cmul(self.noisegpu)
   else
      error('backprop only defined while training')
   end
   return self.gradInput
end

function IndividualDropout:setp(p)
   self.p = torch.FloatTensor(p:nElement())
   self.p:copy(-p):add(1)
   self.pgpu = nil   -- initialized in updateOutput()
end

function IndividualDropout:type(type, tensorCache)
   self.noise = nil  -- make sure this is not copied to GPU
   return Parent.type(self, type, tensorCache)
end
