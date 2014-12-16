-- Copyright 2004-present Facebook. All Rights Reserved.

local async_rng = require('fb.torch.async_rng')
local trace = require('fb.util.trace')

-- Hack: store RNG externally so that we don't try to serialize it...
local rngs = {}

-- Weak keys, so we don't leak RNG objects if the corresponding Dropout
-- objects are destroyed
setmetatable(rngs, {__mode = 'k'})

--[[
A faster variant of `nn.Dropout` that uses the `fblualib` asynchronous RNG.
]]
local Dropout, Parent = torch.class('fbnn.Dropout', 'nn.Module')

--[[
Parameter:

- `p`: the dropout probability (the probability that a given activation will be dropped)
]]
function Dropout:__init(p)
   Parent.__init(self)
   self.p = p or 0.5
   self.train = true
   if self.p >= 1 or self.p < 0 then
      error('<Dropout> illegal percentage, must be 0 <= p < 1')
   end
   self.noise = torch.Tensor()
end

local function concat_tensors(dest, size, tensors)
    local next_index = 1
    dest:resize(size)
    for _, tensor in ipairs(tensors) do
        local n = tensor:nElement()
        dest:narrow(1, next_index, n):copy(tensor)
        next_index = next_index + n
    end
    assert(next_index == dest:nElement() + 1)
end

function Dropout:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if self.train then
       local n = input:nElement()
       local rng = rngs[self]
       if not rng then
           rng = async_rng.bernoulli('torch.FloatTensor', n, 1 - self.p)
           rngs[self] = rng
       end
       local fnoise = rng:generate(n)
       concat_tensors(self.noise, n, fnoise)
       self.noise:resizeAs(input)
       self.output:cmul(self.noise)
   else
       self.output:mul(1-self.p)
   end
   return self.output
end

function Dropout:updateGradInput(input, gradOutput)
   if self.train then
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      -- simply mask the gradients with the noise vector
      self.gradInput:cmul(self.noise)
   else
      error('backprop only defined while training')
   end
   return self.gradInput
end

function Dropout:setp(p)
   self.p = p
   rngs[self] = nil
end
