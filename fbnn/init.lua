require('torch')
require('nn')
require('libfbnn')

pcall(function() include('Dropout.lua') end) -- because uses async_rng
include('Optim.lua')
include('Probe.lua')
include('TrueNLLCriterion.lua')
include('SparseLinear.lua')
include('test.lua')
