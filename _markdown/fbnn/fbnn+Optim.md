

### Optim.lua ###

Copyright 2004-present Facebook. All Rights Reserved.

<a name="fbnn.Optim.dok"></a>


## fbnn.Optim ##


<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/Optim.lua#L28">[src]</a>
<a name="fbnn.Optim.weight_bias_parameters"></a>


### fbnn.Optim.weight_bias_parameters(module) ###

Returns weight parameters and bias parameters and associated grad parameters
for this module. Annotates the return values with flag marking parameter set
as bias parameters set

<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/Optim.lua#L44">[src]</a>
<a name="fbnn.Optim"></a>


### fbnn.Optim(model, optState, checkpoint_data) ###

The regular `optim` package relies on `getParameters`, which is a
beastly abomination before all. This `optim` package uses separate
optim state for each submodule of a `nn.Module`.


#### Undocumented methods ####

<a name="fbnn.Optim:save"></a>
 * `fbnn.Optim:save()`
<a name="fbnn.Optim:type"></a>
 * `fbnn.Optim:type(t)`
<a name="fbnn.Optim:optimize"></a>
 * `fbnn.Optim:optimize(optimMethod, inputs, targets, criterion)`
<a name="fbnn.Optim:setParameters"></a>
 * `fbnn.Optim:setParameters(newParams)`
