local DifferenceOfGaussian, parent =
    torch.class('nn.DifferenceOfGaussian', 'nn.Module')

--[[

This Module implements a layer that performs a difference of Gaussian pyrmiad
filtering of the input. The user needs to specify the number of octaves to
produce (`nOctaves`; default = 3) as well as the number of scales returned per
octave (`nScalesPerOctave`; default = 5).

Example usage:
> im = image.lena()
> nOctaves = 3
> nScalesPerOctave = 5
> net = nn.Sequential()
> net:add(nn.DifferenceOfGaussian(nOctaves, nScalesPerOctave))
> local filteredIm = net:forward(im)

]]--


-- helper function to make Gaussian filter:
local function getGaussianFilter(sigma, sz)
    local sz = sz or math.floor(sigma * 4)
    local filter = torch.Tensor(sz, sz)
    for i = 1,sz do
        for j = 1,sz do
            local D = (i - math.floor(sz / 2)) * (i - math.floor(sz / 2))
                    + (j - math.floor(sz / 2)) * (j - math.floor(sz / 2))
            filter[i][j] = math.exp(-D / (2 * sigma * sigma))
        end
    end
    filter:div(filter:sum())
    return filter
end

-- helper function to subsample an image:
local function subsample(im, stride)
    local sub
    if im:nDimension() == 4 then
        sub = im:index(4, torch.range(stride, im:size(4), stride):long()):index(
            3, torch.range(stride, im:size(3), stride):long()
        )
    else
        sub = im:index(3, torch.range(stride, im:size(3), stride):long()):index(
            2, torch.range(stride, im:size(2), stride):long()
        )
    end
    return sub
end

-- constructor for the layer:
--   * nInputPlane is the number of input planes (channels)
--   * nOctaves is the number of octaves (number of times bandwidth doubles)
--   * nScalesPerOctave is the number of sigma values per octave
--
--   * the output of forward() is a table with nOctaves cell, in which each cell
--     contains a tensor with nScalesPerOctave planes
function DifferenceOfGaussian:__init(nInputPlane, nOctaves, nScalesPerOctave)
    parent.__init(self)
    assert(nInputPlane)
    self.nOctaves = nOctaves or 3
    self.nScalesPerOctave = nScalesPerOctave or 5
    self.nInputPlane = nInputPlane
    self.nOutputPlane = nOctaves
    self.dW = 1
    self.dH = 1

    -- compute sigmas for which to compute filter responses:
    self.sigma = torch.Tensor(self.nScalesPerOctave + 1):fill(1)
    for n = 2,self.nScalesPerOctave do
        self.sigma[n] = self.sigma[n - 1] *
            (math.pow(2, 1 / self.nScalesPerOctave))
    end
    self.sigma[self.nScalesPerOctave + 1] = 2.0
    local sz = math.ceil(self.sigma:max() * 2.5)
    if sz % 2 == 0 then sz = sz + 1 end

    -- construct the Gaussian filters for an octave:
    self.filter = torch.Tensor(self.sigma:nElement(), sz, sz)
    for n = 1,self.sigma:nElement() do
        self.filter[n]:copy(getGaussianFilter(self.sigma[n], sz))
    end

    -- set padding and kernel size:
    self.kH = self.filter[1]:size(1)
    self.kW = self.filter[1]:size(2)
    self.padH = math.floor(self.filter[1]:size(1) / 2)
    self.padW = math.floor(self.filter[1]:size(2) / 2)

    -- initialize buffers for efficiency on GPU:
    self.finput = torch.Tensor()
    self.fgradInput = torch.Tensor()
    self.outputBuf = torch.Tensor()
end

-- function implementing a forward pass in the layer:
function DifferenceOfGaussian:updateOutput(input)

    -- determine which dimension contains channels:
    local batchMode = false
    if input:nDimension() == 4 then
        batchMode = true
    elseif input:nDimension() ~= 3 then
        error('3D or 4D(batch mode) tensor expected')
    end
    local dim = batchMode and 2 or 1
    assert(input:size(dim) == self.nInputPlane)
    local numChannels = self.nInputPlane
    local input = input:clone()

    -- loop over all octaves:
    local finalOutput = {}
    self.output = input.new()
    for m = 1,self.nOctaves do

        -- resize output buffer:
        if batchMode then
            self.outputBuf:resize(
                input:size(1),
                self.sigma:nElement() * numChannels,
                input:size(3),
                input:size(4)
            )
        else
            self.outputBuf:resize(
                self.sigma:nElement() * numChannels,
                input:size(2),
                input:size(3)
            )
        end

        -- apply filters (loop because channels shouldn't be merged):
        self:__prepareFilter(input)
        for c = 1,numChannels do
            local inputPlane = input:narrow(dim, c, 1)
            local output = self.outputBuf:narrow(
                dim, (c - 1) * self.sigma:nElement() + 1,
                self.sigma:nElement())
            input.THNN.SpatialConvolutionMM_updateOutput(
                inputPlane:cdata(),
                output:cdata(),
                self.weight:cdata(),
                self.bias:cdata(),
                self.finput:cdata(),
                self.fgradInput:cdata(),
                self.kW, self.kH,
                self.dW, self.dH,
                self.padW, self.padH
            )
        end

        -- compute the difference of Gaussian responses in-place:
        for c = 1,numChannels do
            self.outputBuf:narrow(
                    dim, (c - 1) * self.sigma:nElement() + 1,
                    self.nScalesPerOctave
            ):add(
                -self.outputBuf:narrow(
                    dim, (c - 1) * self.sigma:nElement() + 2,
                    self.nScalesPerOctave
                )
            )
        end -- leaves the image for sigma = 2 unaltered

        -- set final output of layer:
        local columnInd = torch.LongTensor(self.nScalesPerOctave * numChannels)
        for c = 1,numChannels do
            columnInd:narrow(
                1, (c - 1) * self.nScalesPerOctave + 1,
                self.nScalesPerOctave
            ):copy(
                torch.range(
                    (c - 1) * self.sigma:nElement() + 1,
                    (c - 1) * self.sigma:nElement() + self.nScalesPerOctave
                ):long()
            )   -- we are not storing sigma = 2, will be in next octave
        end
        finalOutput[m] = self.outputBuf:index(dim, columnInd)

        -- subsample blurred input image:
        for c = 1,numChannels do
            input:select(dim, c):copy(
                self.outputBuf:select(
                    dim,
                    (c - 1) * self.sigma:nElement() + self.nScalesPerOctave + 1
                )
            )
        end
        input = subsample(input, 2)
    end

    -- clean up and return:
    self:__cleanStateVars(numChannels)
    self.output = finalOutput
    return self.output
end

-- we do not need to compute parameter gradients as there are no parameters:
function DifferenceOfGaussian:accUpdateGradParameters(input, gradOutput, lr)
end

-- do never update the filters:
function DifferenceOfGaussian:updateParameters(lr)
end

-- this function facilitates different behavior of buffers on CPU and GPU:
function DifferenceOfGaussian:type(type)
    self.finput = torch.Tensor()
    self.fgradInput = torch.Tensor()
    self.outputBuf = torch.Tensor()
    return parent.type(self, type)
end

-- helper function to prepare for filtering:
function DifferenceOfGaussian:__prepareFilter(input)

    -- copy filter here to weights so we can use SpatialConvolutionMM:
    self.weight = input.new(
        self.filter:size(1),
        self.filter:size(2) * self.filter:size(3)
    ):copy(self.filter)
    self.bias = input.new(self.filter:size(1)):zero()
    self.nInputPlane  = 1
    self.nOutputPlane = self.sigma:nElement()
end

-- helper function for cleaning up state variables:
function DifferenceOfGaussian:__cleanStateVars(numChannels)
    self.weight = nil
    self.bias = nil
    self.nInputPlane  = numChannels
    self.nOutputPlane = self.nOctaves
end
