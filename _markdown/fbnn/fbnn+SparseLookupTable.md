

### SparseLookupTable.lua ###

Copyright 2004-present Facebook. All Rights Reserved.

<a name="fbnn.SparseLookupTable.dok"></a>


## fbnn.SparseLookupTable ##


Sparse lookup table. Similar to the regular LookupTable.lua module, 
except for the following differences:

1. The outputs are in sparse format.
2. The inputs are pairs (i,w), so the output corresponding to index i
is scaled by w.
3. The indices are fixed, i.e. during a parameter update only the nonzero 
coefficents are updated. This is to avoid having to create new indices, 
which is expensive and may result in the weights no longer being sparse.


<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/SparseLookupTable.lua#L23">[src]</a>
<a name="fbnn.SparseLookupTable"></a>


### fbnn.SparseLookupTable(indices,sparseGrad) ###


Parameters:
* `indices` is a 2D matrix of indices which will be nonzero.
* `sparseGrad` indicates whether incoming gradients will be sparse or dense.



#### Undocumented methods ####

<a name="fbnn.SparseLookupTable:reset"></a>
 * `fbnn.SparseLookupTable:reset(stdv)`
<a name="fbnn.SparseLookupTable:updateOutput"></a>
 * `fbnn.SparseLookupTable:updateOutput(input)`
<a name="fbnn.SparseLookupTable:accUpdateGradParameters"></a>
 * `fbnn.SparseLookupTable:accUpdateGradParameters(input, gradOutput,lr)`
