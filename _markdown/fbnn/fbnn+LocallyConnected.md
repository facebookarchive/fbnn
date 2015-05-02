

### LocallyConnected.lua ###

Copyright 2004-present Facebook. All Rights Reserved.

<a name="fbnn.LocallyConnected.dok"></a>


## fbnn.LocallyConnected ##


<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/LocallyConnected.lua#L141">[src]</a>
<a name="fbnn.LocallyConnected.toInterleaved"></a>


### fbnn.LocallyConnected.toInterleaved(tensor, make_contiguous) ###

Change a 3-d or 4-d tensor from standard, planar Torch layout (P x H x W) or
(B x P x H x W) to interleaved layout (H x W x P) or (B x H x W x P).
Change a 6-d weight tensor from planar (P_o x H_o x W_o x P_i x H_k x W_k) to
interleaved format (H_o x W_o x H_k x W_k x P_o x P_i). The make_contiguous
flag controls if the result tensor is guaranteed to be contiguous.

<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/LocallyConnected.lua#L169">[src]</a>
<a name="fbnn.LocallyConnected.toPlanar"></a>


### fbnn.LocallyConnected.toPlanar(tensor, make_contiguous) ###

Inverse operation of toInterleaved.

<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/LocallyConnected.lua#L192">[src]</a>
<a name="fbnn.LocallyConnected:type"></a>


### fbnn.LocallyConnected:type(type) ###

Type conversion.
The trick here is to convert the various tensors to and from
cuda layout when a conversion to or from host to GPU takes place.


#### Undocumented methods ####

<a name="fbnn.LocallyConnected"></a>
 * `fbnn.LocallyConnected(nInputPlane, iW, iH, nOutputPlane, kW, kH,
                                 dW, dH)`
<a name="fbnn.LocallyConnected:outputSize"></a>
 * `fbnn.LocallyConnected:outputSize()`
<a name="fbnn.LocallyConnected:reset"></a>
 * `fbnn.LocallyConnected:reset(stdv)`
<a name="fbnn.LocallyConnected:updateOutput"></a>
 * `fbnn.LocallyConnected:updateOutput(input)`
<a name="fbnn.LocallyConnected:updateGradInput"></a>
 * `fbnn.LocallyConnected:updateGradInput(input, gradOutput)`
<a name="fbnn.LocallyConnected:accGradParameters"></a>
 * `fbnn.LocallyConnected:accGradParameters(input, gradOutput, scale)`
