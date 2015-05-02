

### SequentialCriterion.lua ###

Copyright 2004-present Facebook. All Rights Reserved.
Author: Michael Mathieu <myrhev@fb.com>

<a name="fbnn.SequentialCriterion.dok"></a>


## fbnn.SequentialCriterion ##


Combines a module and a criterion.

It is mainly thought for preprocessing, but trainable parameters
can be used if needed



#### Undocumented methods ####

<a name="fbnn.SequentialCriterion"></a>
 * `fbnn.SequentialCriterion(module, criterion)`
<a name="fbnn.SequentialCriterion:parameters"></a>
 * `fbnn.SequentialCriterion:parameters()`
<a name="fbnn.SequentialCriterion:getParameters"></a>
 * `fbnn.SequentialCriterion:getParameters()`
<a name="fbnn.SequentialCriterion:updateOutput"></a>
 * `fbnn.SequentialCriterion:updateOutput(input, target)`
<a name="fbnn.SequentialCriterion:updateGradInput"></a>
 * `fbnn.SequentialCriterion:updateGradInput(input, target)`
<a name="fbnn.SequentialCriterion:accGradParameters"></a>
 * `fbnn.SequentialCriterion:accGradParameters(input, target, scale)`
<a name="fbnn.SequentialCriterion:accUpdateGradParameters"></a>
 * `fbnn.SequentialCriterion:accUpdateGradParameters(input, target, scale)`
<a name="fbnn.SequentialCriterion:updateParameters"></a>
 * `fbnn.SequentialCriterion:updateParameters(learning_rate)`
<a name="fbnn.SequentialCriterion:zeroGradParameters"></a>
 * `fbnn.SequentialCriterion:zeroGradParameters()`
