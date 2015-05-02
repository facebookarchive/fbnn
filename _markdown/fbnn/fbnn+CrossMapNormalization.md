

### CrossMapNormalization.lua ###

Copyright 2004-present Facebook. All Rights Reserved.

<a name="fbnn.CrossMapNormalization.dok"></a>


## fbnn.CrossMapNormalization ##


Cross-map normalization, see
https://code.google.com/p/cuda-convnet/wiki/LayerParams#Local_response_normalization_layer_(across_maps)

formula:

$${f(u_{f}^{x,y})=\frac{u_{f}^{x,y}}{ (1+\frac{\alpha}{N} \sum_{f'=\max(0,F-\lfloor N/2\rfloor )}^{\min(F,f-\lfloor N/2 \rfloor+N) }(u_{f'}^{x,y})^{2})^{\beta}}}$$

where
* ${F}$ is the number of features, 
* ${N}$ is the neighborhood size (size),
* ${\alpha}$ is the scaling factor (scale),
* ${\beta}$ is the exponent (power)

This layer normalizes values across feature maps (each spatial location
independently). Borders are zero-padded.

Parameters:
* `size`: size of the neighborhood (typical value: 5)
* `scale`: scaling factor (typical value: 0.0001)
* `power`: exponent used (typical value: 0.75)



#### Undocumented methods ####

<a name="fbnn.CrossMapNormalization"></a>
 * `fbnn.CrossMapNormalization(size, scale, power)`
<a name="fbnn.CrossMapNormalization:updateOutput"></a>
 * `fbnn.CrossMapNormalization:updateOutput(input)`
<a name="fbnn.CrossMapNormalization:updateGradInput"></a>
 * `fbnn.CrossMapNormalization:updateGradInput(input, gradOutput)`
