require('torch')
require('nn')
require('libfbnn')

pcall(function() include('Dropout.lua') end) -- because uses async_rng
include('UnfoldedTemporalConvolution.lua')
include('NestedDropout.lua')
include('IndividualDropout.lua')
include('CachingLookupTable.lua')
include('Optim.lua')
include('Probe.lua')
include('TrueNLLCriterion.lua')
include('ProjectiveGradientNormalization.lua')
include('ClassNLLCriterionWithUNK.lua')
include('Upsample.lua')
include('test.lua')
include('DataSetLabelMe.lua')
include('LeakyReLU.lua')
include('Constant.lua')
include('SpatialFoveaCuda.lua')
include('SoftPlusLSECriterion.lua')
include('SoftPlusLSEMinusLSECriterion.lua')
-- Former fbcunn.nn_layers
include('ClassHierarchicalNLLCriterion.lua')
include('CrossMapNormalization.lua')
include('GroupKMaxPooling.lua')
include('HSM.lua')
include('KMaxPooling.lua')
include('LinearNB.lua')
include('LaplacianOfGaussian.lua')
include('DifferenceOfGaussian.lua')
include('LinearNoBackprop.lua')
include('LocallyConnected.lua')
include('L2Normalize.lua')
include('NormalizedLinearNoBias.lua')
include('SequentialCriterion.lua')
include('WeightedLookupTable.lua')
include('FasterLookup.lua')

include('SparseNLLCriterion.lua')
include('SparseLinear.lua')
include('SegSparseLinear.lua')
include('SparseThreshold.lua')
local ok, sparse = pcall(require, 'sparse') -- sparse is not oss
if ok then
    include('SparseConverter.lua')
    include('SparseKmax.lua')
    include('SparseLookupTable.lua')
    include('SparseSum.lua')
end

-- Former fbcunn.cpu
local ok, _ = pcall(require,'fbcode.torch.fb.fbnn.cpu_ext')
if not ok then require 'libfbnn' end -- for Open Source
