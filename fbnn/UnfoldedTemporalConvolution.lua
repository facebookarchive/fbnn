local UTC, parent = torch.class('nn.UnfoldedTemporalConvolution', 'nn.Module')

function UTC:__init(nin, nout, kw, dW)
    dW = dW or 1
    assert(dW == 1, "nn.UnfoldedTemporalConvolution only supports dW = 1")

    parent.__init(self)
    self.linear = nn.Linear(kw*nin, nout)

    -- sizes
    self.kW = kw
    self.dW = 1
    self.inputFrameSize = nin
    self.outputFrameSize = nout

    -- expose internal linear parameters
    self.weight = self.linear.weight
    self.bias = self.linear.bias
    self.gradWeight = self.linear.gradWeight
    self.gradBias = self.linear.gradBias

    -- internal buffer
    self._input = torch.Tensor()
    self._unfoldedInput = torch.Tensor()
    self._linearGradOutput = torch.Tensor()
    self._unfoldedGradInput = torch.Tensor()
end

function UTC:noBias()
    self.bias = nil
    self.gradBias = nil
    self.linear:noBias()
    return self
end

function UTC:reset(stdv)
    self.linear:reset(stdv)
end

function UTC:updateOutput(input)
    local nout = self.outputFrameSize
    local nin  = self.inputFrameSize
    local padl = self.kW - 1
    local bsz, l = input:size(1), input:size(2)

    -- no batch?
    if input:dim() == 2 then
        local l = input:size(1)
        self:updateOutput(input:view(1, l, nin))
        self.output = self.output[1]
        return self.output
    end

    -- pad
    -- this a bit complicated but
    -- we want padding at beginning, end and between examples = (bsz + 1) pads
    local n = (bsz + 1) * padl + bsz * l
    self._input
        :resize(n, nin)
        :zero()
        :narrow(1, padl + 1, bsz * (l + padl))
        :view(bsz, -1, nin)
        :narrow(2, 1, l)
        :copy(input)

    -- unfold
    local uinput = self._input
        :view(-1)
        :unfold(1, nin * self.kW, nin)
    self._unfoldedInput
        :resizeAs(uinput)
        :copy(uinput)

    -- linear
    local loutput = self.linear:updateOutput(self._unfoldedInput)
    self.output = loutput
        :view(bsz, -1, nout)
        :narrow(2, padl + 1, l - padl)

    return self.output
end

function UTC:updateGradInput(input, gradOutput)
    local nout = self.outputFrameSize
    local nin  = self.inputFrameSize
    local padl = self.kW - 1
    local bsz, l = input:size(1), input:size(2)

    -- no batch ?
    if input:dim() == 2 then
        local lin, lout = input:size(1), gradOutput:size(1)
        local input = input:view(1, lin, nin)
        local gradOutput = gradOutput:view(1, lout, nout)
        self:updateGradInput(input, gradOutput)
        self.gradInput = self.gradInput[1]
        return self.gradInput
    end

    -- linear
    self._linearGradOutput
        :resizeAs(self.linear.output)
        :zero()
        :view(bsz, -1, nout)
        :narrow(2, padl + 1, l - padl)
        :copy(gradOutput)
    self.linear
        :updateGradInput(self._unfoldedInput, self._linearGradOutput)

    -- reduce
    self._unfoldedGradInput:set(
        self.linear.gradInput:storage(),
        nin * (self.kW - 1) + 1,     -- offset
        bsz,                         -- sz1
        nin * self.kW * (l + padl),  -- st1
        l,                           -- sz2
        nin * self.kW,               -- st2
        self.kW,                     -- sz3
        nin *(self.kW - 1),          -- st3
        nin,                         -- sz4
        1)                           -- st4

    self.gradInput = self.gradInput
        :sum(self._unfoldedGradInput, 3)
        :select(3, 1)

    return self.gradInput
end

function UTC:accGradParameters(input, gradOutput, scale)
    self.linear:accGradParameters(
        self._unfoldedInput, self._linearGradOutput, scale
    )
end

-- we do not need to accumulate parameters when sharing
UTC.sharedAccUpdateGradParameters = UTC.accUpdateGradParameters

function UTC:test()
    local function tensoreq(a, b, epsilon)
        local delta = a:clone():add(-1, b):abs():max()
        return  delta < epsilon, delta
    end

    local function checkforward()
        local input = torch.randn(16, 25, 5)
        local conv = nn.TemporalConvolution(5, 10, 3)
        conv:forward(input)

        local utc = nn.UnfoldedTemporalConvolution(5, 10, 3)
        utc.weight:copy(conv.weight)
        utc.bias:copy(conv.bias)
        utc:forward(input)
        assert(tensoreq(conv.output, utc.output))
    end

    local function checkforwardbackward(bsz, l, nin, nout, kw, type, epsilon)
        local type    = type or 'torch.DoubleTensor'
        local epsilon = epsilon or 1e-12
        local input = torch.randn(bsz, l, nin):type(type)

        local conv = nn.TemporalConvolution(nin, nout, kw):type(type)
        conv:forward(input)
        conv:zeroGradParameters()
        local gout = torch.randn(conv.output:size()):type(type)
        conv:backward(input, gout)

        local utc = nn.UnfoldedTemporalConvolution(nin, nout, kw):type(type)
        utc.weight:copy(conv.weight)
        utc.bias:copy(conv.bias)
        utc:forward(input)
        utc:zeroGradParameters()
        utc:backward(input, gout)

        assert(tensoreq(conv.output, utc.output, epsilon))
        assert(tensoreq(conv.gradInput, utc.gradInput, epsilon))
        assert(tensoreq(conv.gradWeight, utc.gradWeight, epsilon))
        assert(tensoreq(conv.gradBias, utc.gradBias, epsilon))
    end

    return {
        checkforward = checkforward,
        checkforwardbackward = checkforwardbackward,
    }
end
