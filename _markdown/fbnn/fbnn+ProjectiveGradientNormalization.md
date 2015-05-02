<a name="fbnn.ProjectiveGradientNormalization.dok"></a>


## fbnn.ProjectiveGradientNormalization ##

This file implements a projective gradient normalization proposed by Mark Tygert.
   This alters the network from doing true back-propagation.

   The operation implemented is:
   forward:
              Y = X
   backward:
              dL     dL        X      {     X          dL   }
              --  =  --   -  ----  *  |   ----    (.)  --   |
              dX     dY      ||X||    {   ||X||        dY   }
                                  2            2

   where (.) = dot product

   Usage:
   fbnn.ProjectiveGradientNormalization([eps = 1e-5]) -- eps is optional defaulting to 1e-5

   eps is a small value added to the ||X|| to avoid divide by zero
       Defaults to 1e-5


#### Undocumented methods ####

<a name="fbnn.BN"></a>
 * `fbnn.BN(eps)`
<a name="fbnn.BN:updateOutput"></a>
 * `fbnn.BN:updateOutput(input)`
<a name="fbnn.BN:updateGradInput"></a>
 * `fbnn.BN:updateGradInput(input, gradOutput)`
