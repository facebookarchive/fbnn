/**
 * Copyright 2015 Facebook
 */

#include <lua.hpp>
#include <luaT.h>

#include "fblualib/LuaUtils.h"
#include "thpp/Storage.h"
#include "thpp/Tensor.h"

namespace facebook { namespace deeplearning { namespace torch {

using namespace fblualib;
using namespace thpp;

namespace {

template <class T>
int scaleByWeight(lua_State* L) {
  auto output = luaGetTensorChecked<T>(L, 1);
  auto const input = luaGetTensorChecked<T>(L, 2);
  auto const weights = luaGetTensorChecked<T>(L, 3);

  #pragma omp parallel for if(output->size(0) * output->size(1) > 100000)
  for (int i = 0; i < output->size(0); ++i) {
    T weight = weights->at({i});

    T *outputData = output->data() + i * output->stride(0);
    const T *inputData = input->data() + i * input->stride(0);

    for (int j = 0; j < output->size(1); ++j) {
      outputData[j * output->stride(1)] =
        inputData[j * input->stride(1)] * weight;
    }
  }

  return 0;
}

template <class T>
class Registerer {
 private:
  static const luaL_Reg functions_[];

 public:
  static void registerFunctions(lua_State* L);
};

template <class T>
const luaL_Reg Registerer<T>::functions_[] = {
  {"WeightedLookupTable_scaleByWeight", scaleByWeight<T>},
  {nullptr, nullptr},
};

template <class T>
void Registerer<T>::registerFunctions(lua_State* L) {
  luaT_pushmetatable(L, Tensor<T>::kLuaTypeName);
  luaT_registeratname(L, functions_, "nn");
  lua_pop(L, 1);
}

}  // namespace

void initWeightedLookupTable(lua_State* L) {
  Registerer<float>::registerFunctions(L);
  Registerer<double>::registerFunctions(L);
}

}}}  // namespaces
