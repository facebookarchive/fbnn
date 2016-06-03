#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)

#include "src/DataSetLabelMe.c"
#include "THGenerateFloatTypes.h"

#include "src/FasterLookup.c"
#include "THGenerateFloatTypes.h"

#include "src/SparseLinear.c"
#include "THGenerateFloatTypes.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libfbnn(lua_State *L);

int luaopen_libfbnn(lua_State *L)
{
  nn_FloatDataSetLabelMe_init(L);
  nn_DoubleDataSetLabelMe_init(L);
  nn_FloatFasterLookup_init(L);
  nn_DoubleFasterLookup_init(L);
  nn_FloatSparseLinear_init(L);
  nn_DoubleSparseLinear_init(L);

  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "fbnn");
  return 1;
}
