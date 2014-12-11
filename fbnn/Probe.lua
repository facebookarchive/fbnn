local Probe, parent = torch.class('fbnn.Probe', 'nn.Module')

function Probe:__init(module, name)
   -- nn.legacy = true
   if not nn.legacy then
      error('ATM nn.legacy needs to be set to true for this module to work !')
   end

   parent.__init(self)
   self.name = name or 'unnamed'
   -- Use 'modules' in order to specify submodules that get converted to cuda
   self.modules = {}
   self.modules[1] = module
   self:resetTensors()
   self._type = self.modules[1]._type
   nn._ProbeTimer = nn._ProbeTimer or torch.Timer()
end

function Probe:reset(stdv)
   self.modules[1]:reset(stdv)
   self:resetTensors()
end

function Probe:resetTensors()
   if self.modules[1].weight then
      self.weight = self.modules[1].weight
   end
   if self.modules[1].gradWeight then
      self.gradWeight = self.modules[1].gradWeight
   end
   if self.modules[1].bias then
      self.bias = self.modules[1].bias
   end
   if self.modules[1].gradBias then
      self.gradBias = self.modules[1].gradBias
   end
   if self.modules[1].output then
      self.output = self.modules[1].output
   end
   if self.modules[1].gradInput then
      self.gradInput = self.modules[1].gradInput
   end
end

function Probe:setTensors()
   if self.weight then
      self.modules[1].weight = self.weight
   end
   if self.gradWeight then
      self.modules[1].gradWeight = self.gradWeight
   end
   if self.bias then
      self.modules[1].bias = self.bias
   end
   if self.gradBias then
      self.modules[1].gradBias = self.gradBias
   end
   if self.output then
      self.modules[1].output = self.output
   end
   if self.gradInput then
      self.modules[1].gradInput = self.gradInput
   end
end

local function dumpTensorMoments(str, t)
   if t and t:nDimension() > 0 then
      print(str, t:min(), t:max(), t:mean(), t:std(), t:sum())
   end
end

local function dumpTensorOrTableMoments(str, t)
   if torch.type(t) == 'table' then
      for i=1, #t do
         if torch.type(t) == 'torch.IntTensor' then
            dumpTensorMoments(str, t[i]:float())
         else
            dumpTensorMoments(str, t[i])
         end
      end
   else
      if torch.type(t) == 'torch.IntTensor' then
         dumpTensorMoments(str, t:float())
      else
         dumpTensorMoments(str, t)
      end
   end
end

function Probe:dumpModule(name, input, ...)
   print('\n-----------------------------')
   print(name)

   local arg = {...}
   dumpTensorOrTableMoments('module computation input', input)
   for i = 3, #arg do
      dumpTensorOrTableMoments('module computation result ' .. (i - 2), arg[i])
   end

   local m = self.modules[1]
   dumpTensorOrTableMoments('module weight    ', m.weight)
   dumpTensorOrTableMoments('module gradWeight', m.gradWeight)
   dumpTensorOrTableMoments('module bias      ', m.bias)
   dumpTensorOrTableMoments('module gradBias  ', m.gradBias)
   dumpTensorOrTableMoments('module output    ', m.output)
   dumpTensorOrTableMoments('module gradInput ', m.gradInput)
end

function Probe:updateOutput(input)
   self:setTensors()
   self:dumpModule('Start UpdateOutput ' .. self.name, input)
   self.modules[1].output = self.modules[1]:updateOutput(input)
   self:resetTensors()
   self:dumpModule('End UpdateOutput ' .. self.name, input, self.output)
   return self.output
end

function Probe:updateGradInput(input, gradOutput)
   self:setTensors()
   self:dumpModule('Start UpdateGradInput ' .. self.name, gradOutput)
   self.modules[1].gradInput =
      self.modules[1]:updateGradInput(input, gradOutput)
   self:resetTensors()
   self:dumpModule('End UpdateGradInput ' .. self.name,
                   gradOutput,
                   self.gradInput)
   return self.gradInput
end

function Probe:accGradParameters(input, gradOutput, scale)
   self:setTensors()
   self:dumpModule('Start AccGradParameters ' .. self.name, gradOutput)
   self.modules[1]:accGradParameters(input, gradOutput, scale)
   self:resetTensors()
   self:dumpModule('End AccGradParameters ' .. self.name,
                   gradOutput,
                   self.gradWeight,
                   self.gradBias)
end
