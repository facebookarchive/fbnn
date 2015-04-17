require('torch')
require('nn')
require('libfbnn')

pcall(function() include('Dropout.lua') end) -- because uses async_rng
include('CachingLookupTable.lua')
include('Optim.lua')
include('Probe.lua')
include('TrueNLLCriterion.lua')
include('SparseLinear.lua')
include('ProjectiveGradientNormalization.lua')
include('ClassNLLCriterionWithUNK.lua')
include('test.lua')

-- Former fbcunn.nn_layers
include('ClassHierarchicalNLLCriterion.lua')
include('CrossMapNormalization.lua')
include('GroupKMaxPooling.lua')
include('HSM.lua')
include('KMaxPooling.lua')
include('LinearNB.lua')
include('LocallyConnected.lua')
include('SequentialCriterion.lua')
-- include('SparseConverter.lua')
-- include('SparseKmax.lua')
-- include('SparseLookupTable.lua')
-- include('SparseNLLCriterion.lua')
-- include('SparseSum.lua')
-- include('SparseThreshold.lua')
include('WeightedLookupTable.lua')

-- Former fbcunn.cpu
-- require('fbcode.torch.fb.fbnn.cpu_ext')
