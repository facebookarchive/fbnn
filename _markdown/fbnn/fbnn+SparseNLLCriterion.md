

### SparseNLLCriterion.lua ###

Copyright 2004-present Facebook. All Rights Reserved.
Author: Michael Mathieu <myrhev@fb.com>

<a name="fbnn.SparseNLLCriterion.dok"></a>


## fbnn.SparseNLLCriterion ##

Sparse ClassNLL criterion

<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/SparseNLLCriterion.lua#L18">[src]</a>
<a name="fbnn.SparseNLLCriterion"></a>


### fbnn.SparseNLLCriterion(K) ###


Parameters:
* `K` : number of non-zero elements of the target
* `do`_target_check : checks whether the target is a
   probability vector (default true)
* `sizeAverage` : divides the error by the size of the minibatch


<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/SparseNLLCriterion.lua#L38">[src]</a>
<a name="fbnn.SparseNLLCriterion:updateOutput"></a>


### fbnn.SparseNLLCriterion:updateOutput(input, target) ###


`target` should be a table containing two tensors :

```
target = {targetP, targetIdx}
```

where `targetP` are the probabilities associated to the indices `targetIdx`
we assume `targetIdx` doesn't have twice the same number in the same sample.



#### Undocumented methods ####

<a name="fbnn.SparseNLLCriterion:updateGradInput"></a>
 * `fbnn.SparseNLLCriterion:updateGradInput(input, target)`
