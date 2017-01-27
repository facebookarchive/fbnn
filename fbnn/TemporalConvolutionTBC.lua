-- Copyright 2004-present Facebook. All Rights Reserved.

-- Temporal Convolution
-- Input format:
-- Time x BatchSize x Channels

local TBC, parent = torch.class('nn.TemporalConvolutionTBC', 'nn.Module')

function TBC:__init(nIn, nOut, kw,pad)
  pad = pad or 0
  parent.__init(self)

  self.kw = kw
  self.pad = pad
  self.nIn = nIn
  self.nOut = nOut

  self.weight = torch.Tensor(kw,nIn,nOut)
  self.bias = torch.Tensor(nOut)
  self.gradWeight = torch.Tensor(kw,nIn,nOut)
  self.gradBias = torch.Tensor(nOut)
  self:reset()
end

function TBC:reset(stdv)
  if stdv then
    stdv = stdv * math.sqrt(3)
  else
    stdv = 1/math.sqrt(self.kw*self.nIn)
  end
  self.weight:uniform(-stdv, stdv)
  self.bias:uniform(-stdv, stdv)
end

function TBC:noBias()
  assert(false, 'noBias mode not implemented yet!')
end

function TBC:updateOutput(input)
  local s = input:size()
  assert(s:size() == 3)
  assert(s[3] == self.nIn)
  self.output:resize(s[1]-self.kw+1+2*self.pad, s[2], self.nOut)
  input.nn.TemporalConvolutionTBC_updateOutput(self,input)
  return self.output
end

function TBC:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  input.nn.TemporalConvolutionTBC_updateGradInput(self,gradOutput)
  return self.gradInput
end

function TBC:accGradParameters(input, gradOutput, scale)
  scale = scale or 1
  input.nn.TemporalConvolutionTBC_accGradParameters(
    self,input,gradOutput,scale)
end

-- we do not need to accumulate parameters when sharing
TBC.sharedAccUpdateGradParameters = TBC.accUpdateGradParameters

function TBC:clearState()
  return parent.clearState(self)
end

function TBC:test()
  local function tensoreq(a, b, epsilon)
    local delta = a:clone():add(-1, b):abs():max() / torch.abs(a):max()
    print('delta',delta)
    return delta < epsilon, delta
  end

  local function checkforwardbackward(bsz, l, nIn, nOut, kw, pad, type)
    require 'nn'
    require 'cunn'
    type=type or 'torch.FloatTensor'
    bsz = bsz or 64
    l = l or 25
    nIn = nIn or 512
    nOut = nOut or 512
    kw = kw or 3
    local epsilon = (type == 'torch.Tensor') and 1e-14 or 1e-5
    -- random input
    local input = torch.randn(bsz, l, nIn):type(type)

    -- torch reference implementation
    local conv = nn.TemporalConvolution(nIn, nOut, kw, 1)
    local nopad = conv
    if pad then
      conv = nn.Sequential()
        :add(nn.Padding(2, -pad))
        :add(nn.Padding(2, pad))
        :add(conv)
    end
    conv:type(type)
    conv:forward(input)
    conv:zeroGradParameters()
    local gout = torch.randn(conv.output:size()):type(type)
    conv:backward(input, gout)

    -- our implementation
    local tbc = nn.TemporalConvolutionTBC(nIn, nOut, kw, pad)
    tbc:type(type)

    -- adjust weights and input-output format
    input=input:transpose(2,1,3):clone()
    tbc.weight:copy(nopad.weight:reshape(nOut,kw,nIn)
                      :permute(2,3,1):clone())
    gout=gout:transpose(2,1,3):clone()
    tbc.bias:copy(nopad.bias)
    conv.output=conv.output:transpose(2,1,3):clone()
    conv.gradInput=conv.gradInput:transpose(2,1,3):clone()
    nopad.gradWeight=nopad.gradWeight:reshape(nOut,kw,nIn)
      :permute(2,3,1):clone()


    tbc:forward(input)
    tbc:zeroGradParameters()
    tbc:backward(input, gout)

    -- check reference and ours have same outputs, gradients
    assert(tensoreq(conv.output, tbc.output, epsilon))
    assert(tensoreq(nopad.gradBias, tbc.gradBias, epsilon))
    assert(tensoreq(nopad.gradWeight, tbc.gradWeight, epsilon))
    assert(tensoreq(conv.gradInput, tbc.gradInput, epsilon))
  end

  return {
    checkforwardbackward = checkforwardbackward,
  }
end
