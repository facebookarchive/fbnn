

### HSM.lua ###

Copyright 2004-present Facebook. All Rights Reserved.
Author: Michael Mathieu <myrhev@fb.com>

<a name="fbnn.HSM.dok"></a>


## fbnn.HSM ##

Hierarchical soft max with minibatches.

<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/HSM.lua#L24">[src]</a>
<a name="fbnn.HSM"></a>


### fbnn.HSM(mapping, input_size, unk_index) ###


Parameters:
* `mapping` is a table (or tensor) with `n_classes` elements,
    such that `mapping[i]` is a table with 2 elements.
    * `mapping[i][1]` : index (1-based) of the cluster of class `i`
    * `mapping[i][2]` : index (1-based) of the index within its cluster of class `i`
*  `input_size` is the number of elements of the previous layer
*  `unk_index` is an index that is ignored at test time (not added to the
    loss). It can be disabled by setting it to 0 (not nil).
    It should only be used uring testing (since during training,
    it is not disabled in the backprop (TODO) )


<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/HSM.lua#L226">[src]</a>
<a name="fbnn.HSM:updateGradInput"></a>


### fbnn.HSM:updateGradInput(input, target) ###

Note: call this function at most once after each call `updateOutput`,
or the output will be wrong (it uses `class_score` and `cluster_score`
as temporary buffers)

<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/HSM.lua#L270">[src]</a>
<a name="fbnn.HSM:accGradParameters"></a>


### fbnn.HSM:accGradParameters(input, target, scale, direct_update) ###

If `direct_update` is set, the parameters are directly updated (not the
gradients). It means that the gradient tensors (like `cluster_grad_weight`)
are not used. scale must be set to the negative learning rate
(`-learning_rate`). `direct_update` mode is much faster.
Before calling this function you have to call `HSM:updateGradInput` first.


#### Undocumented methods ####

<a name="fbnn.HSM:clone"></a>
 * `fbnn.HSM:clone(...)`
<a name="fbnn.HSM:check_mapping"></a>
 * `fbnn.HSM:check_mapping(mapping)`
<a name="fbnn.HSM:get_n_class_in_cluster"></a>
 * `fbnn.HSM:get_n_class_in_cluster(mapping)`
<a name="fbnn.HSM:parameters"></a>
 * `fbnn.HSM:parameters()`
<a name="fbnn.HSM:getParameters"></a>
 * `fbnn.HSM:getParameters()`
<a name="fbnn.HSM:reset"></a>
 * `fbnn.HSM:reset(weight_stdv, bias_stdv)`
<a name="fbnn.HSM:updateOutput"></a>
 * `fbnn.HSM:updateOutput(input, target)`
<a name="fbnn.HSM:updateOutputCPU"></a>
 * `fbnn.HSM:updateOutputCPU(input, target)`
<a name="fbnn.HSM:updateOutputCUDA"></a>
 * `fbnn.HSM:updateOutputCUDA(input, target)`
<a name="fbnn.HSM:updateGradInputCPU"></a>
 * `fbnn.HSM:updateGradInputCPU(input, target)`
<a name="fbnn.HSM:updateGradInputCUDA"></a>
 * `fbnn.HSM:updateGradInputCUDA(input, target)`
<a name="fbnn.HSM:backward"></a>
 * `fbnn.HSM:backward(input, target, scale)`
<a name="fbnn.HSM:updateParameters"></a>
 * `fbnn.HSM:updateParameters(learning_rate)`
<a name="fbnn.HSM:zeroGradParameters"></a>
 * `fbnn.HSM:zeroGradParameters()`
<a name="fbnn.HSM:zeroGradParametersClass"></a>
 * `fbnn.HSM:zeroGradParametersClass(input, target)`
