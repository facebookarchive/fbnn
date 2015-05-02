

### ClassHierarchicalNLLCriterion.lua ###

Copyright 2004-present Facebook. All Rights Reserved.

<a name="fbnn.ClassHierarchicalNLLCriterion.dok"></a>


## fbnn.ClassHierarchicalNLLCriterion ##


Hierarchical softmax classifier with two levels and arbitrary clusters.

Note:
This criterion does include the lower layer parameters
(this is more `Linear` + `ClassNLLCriterion`, but hierarchical).
Also, this layer does not support the use of mini-batches
(only 1 sample at the time).


<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/ClassHierarchicalNLLCriterion.lua#L25">[src]</a>
<a name="fbnn.ClassHierarchicalNLLCriterion"></a>


### fbnn.ClassHierarchicalNLLCriterion(mapping, clusterCounts, inputSize) ###


Parameters:

* `mapping` is a tensor with as many elements as classes.
   `mapping[i][1]` stores the cluster id, and `mapping[i][2]` the class id within
   that cluster of the ${i}$-th class.
* `clusterCounts` is a vector with as many entry as clusters.
   clusterCounts[i] stores the number of classes in the i-th cluster.
*  `inputSize` is the number of input features


<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/ClassHierarchicalNLLCriterion.lua#L66">[src]</a>
<a name="fbnn.ClassHierarchicalNLLCriterion:updateOutput"></a>


### fbnn.ClassHierarchicalNLLCriterion:updateOutput(input, target) ###

`target` is the class id

<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/ClassHierarchicalNLLCriterion.lua#L112">[src]</a>
<a name="fbnn.ClassHierarchicalNLLCriterion:updateGradInput"></a>


### fbnn.ClassHierarchicalNLLCriterion:updateGradInput(input, target) ###

This computes derivatives w.r.t. input and parameters.

<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/ClassHierarchicalNLLCriterion.lua#L138">[src]</a>
<a name="fbnn.ClassHierarchicalNLLCriterion:updateParameters"></a>


### fbnn.ClassHierarchicalNLLCriterion:updateParameters(learningRate) ###

Update parameters (only those that are used to process this sample).

<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/ClassHierarchicalNLLCriterion.lua#L150">[src]</a>
<a name="fbnn.sampleMultiNomial"></a>


### fbnn.sampleMultiNomial(input) ###

input is a vector of probabilities (non-negative and sums to 1).

<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/ClassHierarchicalNLLCriterion.lua#L164">[src]</a>
<a name="fbnn.ClassHierarchicalNLLCriterion:infer"></a>


### fbnn.ClassHierarchicalNLLCriterion:infer(input, sampling) ###

Inference of the output (to be used at test time only)
If sampling flag is set to true, then the output label is sampled
o/w the most likely class is provided.

<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/ClassHierarchicalNLLCriterion.lua#L202">[src]</a>
<a name="fbnn.ClassHierarchicalNLLCriterion:eval"></a>


### fbnn.ClassHierarchicalNLLCriterion:eval(input, target) ###

Given some label, it computes the logprob and the ranking error.


#### Undocumented methods ####

<a name="fbnn.ClassHierarchicalNLLCriterion:zeroGradParameters"></a>
 * `fbnn.ClassHierarchicalNLLCriterion:zeroGradParameters()`
<a name="fbnn.ClassHierarchicalNLLCriterion:zeroGradParametersCluster"></a>
 * `fbnn.ClassHierarchicalNLLCriterion:zeroGradParametersCluster()`
<a name="fbnn.ClassHierarchicalNLLCriterion:zeroGradParametersClass"></a>
 * `fbnn.ClassHierarchicalNLLCriterion:zeroGradParametersClass(target)`
