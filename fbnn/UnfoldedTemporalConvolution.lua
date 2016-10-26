local UTC, parent = torch.class('nn.UnfoldedTemporalConvolution', 'nn.Module')

function UTC:__init(nin, nout, kw, dW, pad)
    dW = dW or 1
    assert(dW == 1, "nn.UnfoldedTemporalConvolution only supports dW = 1")

    parent.__init(self)
    self.linear = nn.Linear(kw*nin, nout)

    -- sizes
    self.kW = kw
    self.dW = 1
    self.pad = pad or 0
    self.inputFrameSize = nin
    self.outputFrameSize = nout

    -- expose internal linear parameters
    self.weight = self.linear.weight
    self.bias = self.linear.bias
    self.gradWeight = self.linear.gradWeight
    self.gradBias = self.linear.gradBias

    -- internal buffer
    self._paddedInput = torch.Tensor()
    self._paddedGradInput = torch.Tensor()
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
    local kpad = self.kW - 1    -- pad added between sentence in batch
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
    local padl = l + 2 * self.pad           -- padded length
    local n = (bsz + 1) * kpad + bsz * padl -- add kpad between each sample
    self._paddedInput
        :resize(n, nin)
        :zero()
        :narrow(1, kpad + 1, bsz * (padl + kpad))
        :view(bsz, -1, nin)
        :narrow(2, 1 + self.pad, l)
        :copy(input)

    -- unfold
    local uinput = self._paddedInput
        :view(-1)
        :unfold(1, nin * self.kW, nin)
    self._unfoldedInput
        :resizeAs(uinput)
        :copy(uinput)

    -- linear
    local loutput = self.linear:updateOutput(self._unfoldedInput)
    self.output = loutput
        :view(bsz, -1, nout)
        :narrow(2, kpad + 1, padl - kpad)

    return self.output
end

function UTC:updateGradInput(input, gradOutput)
    local nout = self.outputFrameSize
    local nin  = self.inputFrameSize
    local kpad = self.kW - 1
    local bsz, l = input:size(1), input:size(2)
    local padl = l + 2*self.pad

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
        :narrow(2, kpad + 1, padl - kpad)
        :copy(gradOutput)
    self.linear
        :updateGradInput(self._unfoldedInput, self._linearGradOutput)

    -- reduce
    self._unfoldedGradInput:set(
        self.linear.gradInput:storage(),
        nin * (self.kW - 1) + 1,        -- offset
        bsz,                            -- sz1
        nin * self.kW * (padl + kpad),  -- st1
        padl,                           -- sz2
        nin * self.kW,                  -- st2
        self.kW,                        -- sz3
        nin *(self.kW - 1),             -- st3
        nin,                            -- sz4
        1)                              -- st4

    self._paddedGradInput = self._paddedGradInput
        :sum(self._unfoldedGradInput, 3)
        :select(3, 1)

    self.gradInput = self._paddedGradInput
        :narrow(2, 1 + self.pad, l)

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

    local function checkforwardbackward(bsz, l, nin, nout, kw, pad, type, eps)
        bsz = bsz or 16
        l = l or 25
        nin = nin or 5
        nout = nout or 10
        kw = kw or 3
        type    = type or 'torch.DoubleTensor'
        local epsilon = eps or 1e-12

        -- random input
        local input = torch.randn(bsz, l, nin):type(type)

        -- torch reference implementation
        local conv = nn.TemporalConvolution(nin, nout, kw, 1)
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
        local utc = nn.UnfoldedTemporalConvolution(nin, nout, kw, 1, pad)
        utc:type(type)
        utc.weight:copy(nopad.weight)
        utc.bias:copy(nopad.bias)
        utc:forward(input)
        utc:zeroGradParameters()
        utc:backward(input, gout)

        -- check reference and ours have same outputs, gradients
        assert(tensoreq(conv.output, utc.output, epsilon))
        assert(tensoreq(conv.gradInput, utc.gradInput, epsilon))
        assert(tensoreq(nopad.gradWeight, utc.gradWeight, epsilon))
        assert(tensoreq(nopad.gradBias, utc.gradBias, epsilon))
    end

    return {
        checkforwardbackward = checkforwardbackward,
    }
end
