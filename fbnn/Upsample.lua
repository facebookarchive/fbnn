local Upsample, parent = torch.class('nn.Upsample', 'nn.Module')

function Upsample:__init(factor)
    assert(factor == math.floor(factor), 'Upsampling factor must be integer.')
    assert(factor >= 1, 'Upsampling factor cannot be smaller than 1.')
    parent.__init(self)
    self.factor = factor
    self.buffer = torch.Tensor()
end

function Upsample:updateOutput(input)
    if self.factor ~= 1 then

        -- set some variables:
        local batchMode = (input:nDimension() == 4)
        local offset = batchMode and 1 or 0
        local channels = input:size(1 + offset)
        local height   = input:size(2 + offset)
        local width    = input:size(3 + offset)

        -- compute sizes for the magic incantation:
        local extendedSize, expandedSize
        if batchMode then
            self.output:resize(
                input:size(1),
                channels,
                height * self.factor,
                width  * self.factor
            )
            extendedSize = torch.LongStorage(
                {input:size(1), channels, height, width, 1, 1}
            )
            expandedSize = torch.LongStorage(
                {input:size(1), channels, height, width, self.factor,
                    self.factor}
            )
        else
            self.output:resize(
                channels,
                height * self.factor,
                width * self.factor
            )
            extendedSize = torch.LongStorage(
                {channels, height, width, 1, 1}
            )
            expandedSize = torch.LongStorage(
                {channels, height, width, self.factor, self.factor}
            )
        end

        -- perform upsampling without loops in a single copy:
        local inputView =
            input:contiguous():view(extendedSize):expand(expandedSize)
        inputView = inputView:transpose(3 + offset, 4 + offset)
        self.output:viewAs(inputView):copy(inputView)
    else
        self.output:resizeAs(input):copy(input)
    end
    return self.output
end

function Upsample:updateGradInput(input, gradOutput)
    if self.gradInput then
        if self.factor ~= 1 then

            -- set some variables:
            local batchMode = (input:nDimension() == 4)
            local offset = batchMode and 1 or 0
            local channels = input:size(1 + offset)
            local height   = input:size(2 + offset)
            local width    = input:size(3 + offset)

            -- compute size for the magic incantation:
            local viewSize, sumSize
            if batchMode then
                viewSize = torch.LongStorage(
                    {input:size(1), channels, height, self.factor, width,
                        self.factor}
                )
                sumSize = torch.LongStorage(
                    {input:size(1), channels, height, width,
                        self.factor * self.factor}
                )
            else
                viewSize = torch.LongStorage(
                    {channels, height, self.factor, width, self.factor}
                )
                sumSize = torch.LongStorage(
                    {channels, height, width, self.factor * self.factor}
                )
            end

            -- perform "downsumming" without loops and a single copy:
            local gradView = gradOutput:view(viewSize):transpose(
                3 + offset, 4 + offset
            ):contiguous():view(sumSize)
            self.gradInput:sum(gradView, 4 + offset):resizeAs(input)
        else
            self.gradInput:resizeAs(gradOutput):copy(gradOutput)
        end
    end
end

function Upsample:__tostring__()
    return torch.type(self) .. string.format('(factor %d)', self.factor)
end
