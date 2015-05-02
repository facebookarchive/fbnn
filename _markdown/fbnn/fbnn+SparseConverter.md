<a name="fbnn.SparseConverter.dok"></a>


## fbnn.SparseConverter ##

Copyright 2004-present Facebook. All Rights Reserved.

<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/SparseConverter.lua#L14">[src]</a>
<a name="fbnn.SparseConverter"></a>


### fbnn.SparseConverter(fconv,bconv,dim,thresh) ###


Parameters:
* `fconv` - conversion to perform in fprop, either 'StoD','DtoS' or nil
* `bconv` - conversion to perform in bprop, either 'StoD','DtoS' or nil
* `dim` - number of dimensions
* `thresh` - threshold for sparsifying (0 by default)



#### Undocumented methods ####

<a name="fbnn.SparseConverter:updateOutput"></a>
 * `fbnn.SparseConverter:updateOutput(input)`
<a name="fbnn.SparseConverter:updateGradInput"></a>
 * `fbnn.SparseConverter:updateGradInput(input, gradOutput)`
