local NestedDropout, Parent = torch.class('fbnn.NestedDropout', 'nn.Dropout')

--[[

This implements the nested dropout layer described in the paper:

   O. Rippel, M.A. Gelbart, and R.P. Adams. Learning Ordered Representations
   with Nested Dropout. ICML 2014.

The layer applies a different amount of dropout to each of the inputs.
In practice, it samples a value b and drops units b+1,...,K. The value b is
drawn from the geometric distribution p(b) = p^{b-1}(1-p) with free parameter p.

]]--

function NestedDropout:__init(p)
   Parent.__init(self)
   self.p = p or 0.01
   self.train = true
   if self.p >= 1 or self.p <= 0 then
      error('<NestedDropout> illegal geometric distribution parameter, ' ..
         'must be 0 < p < 1')
   end
   self.logp = math.log(p)
   self.log1p = math.log(1 - p)
   self.noise = torch.Tensor()
end

function NestedDropout:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if self.train then
      self.noise:resizeAs(input)
      self.noise:fill(1)
      local normalization = torch.range(1, self.noise:size(2))
      normalization:add(-1):mul(self.log1p):add(self.logp):exp()
      normalization = -normalization:cumsum():add(-1)
      for n = 1,self.noise:size(1) do
         local b = torch.geometric(1 - self.p)
         if b < self.noise:size(2) then
           self.noise[n]:narrow(1, b + 1, self.noise:size(2) - b):zero()
         end
      end
      normalization = normalization:reshape(1, self.noise:size(2))
      normalization = normalization:expandAs(self.noise)
      self.noise:cdiv(normalization)
      self.output:cmul(self.noise)  -- This is inefficient!
   end
   return self.output
end

function NestedDropout:updateGradInput(input, gradOutput)
   if self.train then
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      self.gradInput:cmul(self.noise)  -- This is inefficient!
   else
      error('backprop only defined while training')
   end
   return self.gradInput
end

function NestedDropout:setp(p)
   self.p = p
end
