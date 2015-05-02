

### WeightedLookupTable.lua ###

Copyright 2004-present Facebook. All Rights Reserved.

<a name="fbnn.WeightedLookupTable.dok"></a>


## fbnn.WeightedLookupTable ##


<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/WeightedLookupTable.lua#L52">[src]</a>
<a name="fbnn.WeightedLookupTable:updateOutput"></a>


### fbnn.WeightedLookupTable:updateOutput(input) ###


Parameters:
* `Input` should be an n x 2 tensor where the first column is dictionary indexes
   and the second column is weights.



#### Undocumented methods ####

<a name="fbnn.WeightedLookupTable"></a>
 * `fbnn.WeightedLookupTable(nIndex, ...)`
<a name="fbnn.WeightedLookupTable:reset"></a>
 * `fbnn.WeightedLookupTable:reset(stdv)`
<a name="fbnn.WeightedLookupTable:zeroGradParameters"></a>
 * `fbnn.WeightedLookupTable:zeroGradParameters()`
<a name="fbnn.WeightedLookupTable:accGradParameters"></a>
 * `fbnn.WeightedLookupTable:accGradParameters(input, gradOutput, scale)`
<a name="fbnn.WeightedLookupTable:accUpdateGradParameters"></a>
 * `fbnn.WeightedLookupTable:accUpdateGradParameters(input, gradOutput, lr)`
<a name="fbnn.WeightedLookupTable:updateParameters"></a>
 * `fbnn.WeightedLookupTable:updateParameters(learningRate)`
