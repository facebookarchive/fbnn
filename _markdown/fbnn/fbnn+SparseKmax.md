

### SparseKmax.lua ###

Copyright 2004-present Facebook. All Rights Reserved.

<a name="fbnn.SparseKmax.dok"></a>


## fbnn.SparseKmax ##


This module performs a sparse embedding with the following process:

1. Perform a dense embedding
2. Apply a linear transformation (to high dimensional space)
3. Make the output k-sparse

The parameters of the dense embedding and the linear transformation are 
learned. Since the fprop may be slow, we keep a candidate set for each word
which consists of the most likely indices to be turned on after the k-max
operation. We record the number of activations for each member of this set,
and periodically resize it to keep only the most active indices. 
Thus the initial training with large candidate sets will be slow, but will
get faster and faster as we restrict their sizes.


<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/SparseKmax.lua#L31">[src]</a>
<a name="fbnn.SparseKmax"></a>


### fbnn.SparseKmax(vocabSize,nDenseDim,nSparseDim,k,nCandidates) ###


Parameters:
* `vocabSize` - number of entries in the dense lookup table
* `nDenseDim` - number of dimensions for initial dense embedding
* `nSparseDim` - number of dimensions for final sparse embedding
* `k` - number of nonzeros in sparse space
* `nCandidates` - initial size of the candidate set


<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/SparseKmax.lua#L93">[src]</a>
<a name="fbnn.SparseKmax:updateCandidateSets"></a>


### fbnn.SparseKmax:updateCandidateSets(nCandidates) ###

Update the candidate set based on the counts of activations.
`nCandidates` is the size of the new candidate sets.

<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/SparseKmax.lua#L110">[src]</a>
<a name="fbnn.SparseKmax:accUpdateGradParameters"></a>


### fbnn.SparseKmax:accUpdateGradParameters(input, gradOutput, lr) ###

Note, we assume `gradOutput` is sparse since the output is sparse as well.


#### Undocumented methods ####

<a name="fbnn.SparseKmax:reset"></a>
 * `fbnn.SparseKmax:reset(stdv)`
<a name="fbnn.SparseKmax:updateOutput"></a>
 * `fbnn.SparseKmax:updateOutput(input)`
