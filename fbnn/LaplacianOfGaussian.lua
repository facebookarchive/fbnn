local LaplacianOfGaussian, parent =
    torch.class('nn.LaplacianOfGaussian', 'nn.Module')

--[[

This Module implements a layer that performs a Laplacian of Gaussian filtering
of the input at a particular scale, which is defined by the filter bandwidth
sigma. The parameter sigma is defined in terms of pixels (note that this is
different from the definition of sigma in image.gaussian, where sigma is defined
relative to the size of the filter). By default, the used filter size will be
10 times sigma, rounded to the nearest odd number. Also, the input will be
padded by default such that the output size equals the input size.
Alternatively, the filter size and the padding can be specified manually.

Example usage:
> im = image.lena()
> net = nn.Sequential()
> net:add(nn.LaplacianOfGaussian(1))
> local filteredIm = net:forward(im)

]]--

-- function that generates a Laplacian of Gaussian filter:
local function getFilter(sigma, sz)

    -- defaults:
    local sz = sz or math.ceil(sigma * 10)
    if sz % 2 == 0 then sz = sz + 1 end

    -- generate Gaussian kernel:
    local image = require 'image'
    local ker = image.gaussian{normalize = true,
                               width  = sz,
                               height = sz,
                               sigma = .1}

    -- compute first derivatives in both directions:
    local dKdY = torch.add(ker:narrow(1, 1, ker:size(1) - 1),
                          -ker:narrow(1, 2, ker:size(1) - 1))
    local dKdX = torch.add(ker:narrow(2, 1, ker:size(2) - 1),
                          -ker:narrow(2, 2, ker:size(2) - 1))

    -- compute second derivatives in both directions:
    local dKdYY = torch.add(dKdY:narrow(1, 1, dKdY:size(1) - 1),
                           -dKdY:narrow(1, 2, dKdY:size(1) - 1))
    local dKdXX = torch.add(dKdX:narrow(2, 1, dKdX:size(2) - 1),
                           -dKdX:narrow(2, 2, dKdX:size(2) - 1))

    -- compute final Laplacian of Gaussian kernel:
    local LoG = torch.add(dKdYY:narrow(2, 2, dKdXX:size(2)),
                          dKdXX:narrow(1, 2, dKdYY:size(1)))

    -- return filter:
    return LoG
end

-- function that initializes the layer:
--   * nInputPlane is the number of input planes
--   * sigma is the filter bandwidth in pixels
--   * k is the filter size (square filters only; default = 10 * sigma)
--   * padding is the padding (on both sides; default is half the filter size)
function LaplacianOfGaussian:__init(nInputPlane, sigma, k, padding)
    local filter = getFilter(sigma, k)
    self.nInputPlane = nInputPlane
    self.nOutputPlane = nInputPlane
    self.weight = torch.Tensor(self.nInputPlane, filter:size(1), filter:size(2))
    for c = 1,self.nInputPlane do
        self.weight[c]:copy(filter)
    end
    self.bias = self.weight.new(1):zero()
    self.padding = padding or math.floor(filter:size(1) / 2)
    self.kH = filter:size(1)
    self.kW = filter:size(2)
    self.dW = 1
    self.dH = 1
    self.connTable = nn.tables.oneToOne(self.nInputPlane)
    self.output = torch.Tensor()
    self.finput = torch.Tensor()
    self.fgradInput = torch.Tensor()
    self:reset()
end

function LaplacianOfGaussian:reset()
end

-- function implementing a forward pass in the layer:
function LaplacianOfGaussian:updateOutput(input)

    -- assertions on input:
    local dim
    if input:nDimension() == 3 then dim = 1
    elseif input:nDimension() == 4 then dim = 2
    else error('3D or 4D(batch mode) tensor expected') end
    assert(input:size(dim) == self.nInputPlane)

    -- perform padding (SpatialConvolutionMap doesn't support this):
    local paddedInput
    local padding = self.padding
    if padding > 0 then
        if dim == 1 then
            paddedInput = input.new(input:size(1), input:size(2) + 2 * padding,
                                                   input:size(3) + 2 * padding)
            paddedInput:sub(1, self.nInputPlane,
                            padding + 1, padding + input:size(2),
                            padding + 1, padding + input:size(3)):copy(input)
        else
            paddedInput = input.new(input:size(1), input:size(2),
                                                   input:size(3) + 2 * padding,
                                                   input:size(4) + 2 * padding)
            paddedInput:sub(1, input:size(1), 1, self.nInputPlane,
                            padding + 1, padding + input:size(2),
                            padding + 1, padding + input:size(3)):copy(input)
        end
    else
        paddedInput = input
    end

    -- apply filter to each channel separately:
    self.output:resizeAs(input)
    input.THNN.SpatialConvolutionMap_updateOutput(
      paddedInput:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      self.connTable:cdata(),
      self.nInputPlane,
      self.nOutputPlane,
      self.dW, self.dH
    )

    -- return:
    return self.output
end

-- function implementing a backward pass in the layer:
function LaplacianOfGaussian:updateGradInput(input, gradOutput)
    self.gradInput = input.new(1)
    --if self.gradInput then

        -- determine which dimension contains channels:
        local dim
        if input:nDimension() == 3 then dim = 1
        elseif input:nDimension() == 4 then dim = 2
        else error('3D or 4D(batch mode) tensor expected') end
        assert(input:size(dim) == self.nInputPlane)

        -- perform padding (SpatialConvolutionMap doesn't support this):
        local paddedInput
        local padding = self.padding
        if padding > 0 then
            if dim == 1 then
                paddedInput = input.new(input:size(1),
                                        input:size(2) + 2 * padding,
                                        input:size(3) + 2 * padding)
                paddedInput:sub(1, self.nInputPlane,
                        padding + 1, padding + input:size(2),
                        padding + 1, padding + input:size(3)):copy(input)
            else
                paddedInput = input.new(input:size(1),
                                        input:size(2),
                                        input:size(3) + 2 * padding,
                                        input:size(4) + 2 * padding)
                paddedInput:sub(1, input:size(1), 1, self.nInputPlane,
                        padding + 1, padding + input:size(2),
                        padding + 1, padding + input:size(3)):copy(input)
            end
        else
            paddedInput = input
        end

        -- perform backwards pass for each channel separately:
        local gradInputBuf = gradOutput.new(paddedInput:size())
        input.THNN.SpatialConvolutionMap_updateGradInput(
           paddedInput:cdata(),
           gradOutput:cdata(),
           gradInputBuf:cdata(),
           self.weight:cdata(),
           self.bias:cdata(),
           self.connTable:cdata(),
           self.nInputPlane,
           self.nOutputPlane,
           self.dW, self.dH
        )

        -- remove padding and return:
        if dim == 1 then
            self.gradInput:resizeAs(input):copy(
                gradInputBuf:sub(1, self.nInputPlane,
                                 padding + 1, padding + input:size(2),
                                 padding + 1, padding + input:size(3))
            )
        else
            self.gradInput:resizeAs(input):copy(
                gradInputBuf:sub(1, input:size(1), 1, self.nInputPlane,
                                 padding + 1, padding + input:size(3),
                                 padding + 1, padding + input:size(4))
            )
        end
    --end
    return self.gradInput
end

-- we do not need to compute gradients as there are no parameters:
function LaplacianOfGaussian:accUpdateGradParameters(input, gradOutput, lr)
end

-- do never update the filters:
function LaplacianOfGaussian:updateParameters(lr)
end

-- this function facilitates different behavior of buffers on CPU and GPU:
function LaplacianOfGaussian:type(type)
   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()
   return parent.type(self, type)
end
