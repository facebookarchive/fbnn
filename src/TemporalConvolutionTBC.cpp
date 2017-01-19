// Copyright 2004-present Facebook. All Rights Reserved.
// Author: Benjamin Graham <benjamingraham@fb.com>

// Tensor formats
// Input: ilen * batchSize * inputPlanes
// Output: olen * batchSize * outputPlanes
// Weight: kw * inputPlanes * outputPlanes

#include <lua.hpp>
#include <luaT.h>
#include <mkl.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include "Blas.h"
#include "Vml.h"
#include "fblualib/LuaUtils.h"
#include "thpp/Storage.h"
#include "thpp/Tensor.h"

namespace facebook {
namespace deeplearning {
namespace torch {

using namespace fblualib;
using namespace thpp;

namespace {

template <class T>
using thOps = thpp::detail::TensorOps<T>;

template <class T>
int updateOutput(lua_State* L) {
  auto output = luaGetFieldIfTensorChecked<T>(L, 1, "output");
  auto weight = luaGetFieldIfTensorChecked<T>(L, 1, "weight");
  auto bias = luaGetFieldIfTensorChecked<T>(L, 1, "bias");
  auto input = luaGetTensorChecked<T>(L, 2);
  auto ilen = input->size(0);
  auto batchSize = input->size(1);
  auto inputPlanes = input->size(2);
  auto outputPlanes = output->size(2);
  auto olen = output->size(0);
  auto kw = weight->size(0);
  int pad = (olen - ilen + kw - 1) / 2;

  luaL_argcheck(
      L,
      (output->ndims() == 3) && (output->size(1) == batchSize) &&
          (olen == ilen - kw + 1 + 2 * pad),
      1,
      "output has wrong dimension");
  luaL_argcheck(L, (input->ndims() == 3), 2, "input has wrong dimension");
  luaL_argcheck(
      L,
      (weight->ndims() == 3) && (weight->size(1) == inputPlanes) &&
          (weight->size(2) == outputPlanes),
      1,
      "weight has wrong dimension");
  luaL_argcheck(
      L,
      (bias->ndims() == 1) && (bias->size(0) == outputPlanes),
      1,
      "bias has wrong dimension");

  auto W = weight->data();
  auto B = bias->data();
  auto I = input->data();
  auto O = output->data();

  for (int t = 0; t < olen; t++)
    for (int b = 0; b < batchSize; b++)
      for (int c = 0; c < outputPlanes; c++)
        O[t * output->stride(0) + b * output->stride(1) + c] = B[c];
  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - pad);
    int oShift = std::max(0, pad - k);
    int t = std::min(ilen + pad - k, olen) - oShift;
    // Note: using gemm in column-major order mode
    // input    is l*m (row-major)
    // weight   is m*r (row-major)
    // output   is l*r (row-major)
    if (t > 0)
      blas::gemm(
          CblasColMajor,
          CblasNoTrans,
          CblasNoTrans,
          outputPlanes, // r
          batchSize * t, // l
          inputPlanes, // m
          1, // alpha
          W + k * weight->stride(0),
          outputPlanes,
          I + iShift * input->stride(0),
          input->stride(1),
          1, // beta
          O + oShift * output->stride(0),
          output->stride(1)
          );
  }
  return 0;
}
template <class T>
int updateGradInput(lua_State* L) {
  auto dInput = luaGetFieldIfTensorChecked<T>(L, 1, "gradInput");
  auto weight = luaGetFieldIfTensorChecked<T>(L, 1, "weight");
  auto dOutput = luaGetTensorChecked<T>(L, 2);
  auto ilen = dInput->size(0);
  auto batchSize = dInput->size(1);
  auto inputPlanes = dInput->size(2);
  auto outputPlanes = dOutput->size(2);
  auto olen = dOutput->size(0);
  auto kw = weight->size(0);
  int pad = (olen - ilen + kw - 1) / 2;

  auto W = weight->data();
  auto dI = dInput->data();
  auto dO = dOutput->data();

  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - pad);
    int oShift = std::max(0, pad - k);
    int t = std::min(ilen + pad - k, olen) - oShift;
    // dOutput * T(weight) -> dInput
    // Note: using gemm in column-major order mode
    // dOutput is l*m (row-major)
    // weight  is r*m (row-major)
    // dInput  is l*r (row-major)
    if (t > 0)
      blas::gemm(
          CblasColMajor,
          CblasTrans,
          CblasNoTrans,
          inputPlanes, // r
          batchSize * t, // l
          outputPlanes, // m
          1, // alpha
          W + k * weight->stride(0),
          outputPlanes,
          dO + oShift * dOutput->stride(0),
          dOutput->stride(1),
          1, // beta
          dI + iShift * dInput->stride(0),
          dInput->stride(1)
          );
  }
  return 0;
}

template <class T>
int accGradParameters(lua_State* L) {
  auto dWeight = luaGetFieldIfTensorChecked<T>(L, 1, "gradWeight");
  auto dBias = luaGetFieldIfTensorChecked<T>(L, 1, "gradBias");
  auto input = luaGetTensorChecked<T>(L, 2);
  auto dOutput = luaGetTensorChecked<T>(L, 3);
  T scale = luaGetNumberChecked<T>(L, 4);
  auto ilen = input->size(0);
  auto batchSize = input->size(1);
  auto inputPlanes = input->size(2);
  auto outputPlanes = dOutput->size(2);
  auto olen = dOutput->size(0);
  auto kw = dWeight->size(0);
  int pad = (olen - ilen + kw - 1) / 2;

  auto dW = dWeight->data();
  auto dB = dBias->data();
  auto I = input->data();
  auto dO = dOutput->data();

  for (int t = 0; t < olen; t++)
    for (int b = 0; b < batchSize; b++)
      for (int c = 0; c < outputPlanes; c++)
        dB[c] += dO[t * dOutput->stride(0) + b * dOutput->stride(1) + c];

  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - pad);
    int oShift = std::max(0, pad - k);
    int t = std::min(ilen + pad - k, olen) - oShift;
    // Note: using gemm in column-major order mode
    // Input    is m*l (row-major)
    // dOutput  is m*r (row-major)
    // dWeight  is l*r (row-major)
    if (t > 0)
      blas::gemm(
          CblasColMajor,
          CblasNoTrans,
          CblasTrans,
          outputPlanes, // r
          inputPlanes, // l
          batchSize * t, // m
          scale, // alpha
          dO + oShift * dOutput->stride(0),
          dOutput->stride(1),
          I + iShift * input->stride(0),
          input->stride(1),
          1, // beta
          dW + k * dWeight->stride(0),
          outputPlanes
          );
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
    {"TemporalConvolutionTBC_updateOutput", updateOutput<T>},
    {"TemporalConvolutionTBC_updateGradInput", updateGradInput<T>},
    {"TemporalConvolutionTBC_accGradParameters", accGradParameters<T>},
    {nullptr, nullptr},
};

template <class T>
void Registerer<T>::registerFunctions(lua_State* L) {
  luaT_pushmetatable(L, Tensor<T>::kLuaTypeName);
  luaT_registeratname(L, functions_, "nn");
  lua_pop(L, 1);
}

} // namespace

void initTemporalConvolutionTBC(lua_State* L) {
  Registerer<float>::registerFunctions(L);
  Registerer<double>::registerFunctions(L);
}
}
}
} // namespaces
