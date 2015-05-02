

### Dropout.lua ###

Copyright 2004-present Facebook. All Rights Reserved.

<a name="fbnn.Dropout.dok"></a>


## fbnn.Dropout ##


A faster variant of `nn.Dropout` that uses the `fblualib` asynchronous RNG.


<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/Dropout.lua#L23">[src]</a>
<a name="fbnn.Dropout"></a>


### fbnn.Dropout(p) ###


Parameter:

- `p`: the dropout probability (the probability that a given activation will be dropped)



#### Undocumented methods ####

<a name="fbnn.Dropout:updateOutput"></a>
 * `fbnn.Dropout:updateOutput(input)`
<a name="fbnn.Dropout:updateGradInput"></a>
 * `fbnn.Dropout:updateGradInput(input, gradOutput)`
<a name="fbnn.Dropout:setp"></a>
 * `fbnn.Dropout:setp(p)`
