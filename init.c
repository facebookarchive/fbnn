#include "TH.h"
#include "luaT.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libfbnn(lua_State *L);

int luaopen_libfbnn(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "fbnn");

  return 1;
}

