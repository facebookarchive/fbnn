#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "src/SparseLinear.c"
#else

#ifdef _OPENMP
#include <omp.h>
#endif

#define ROW_PTR2(t, r) (THTensor_(data)(t) + (r) * (t)->stride[0])
#define COL_PTR2(t, c) (THTensor_(data)(t) + (c) * (t)->stride[1])

static int nn_(checkInput)(THTensor* t) {
  return t->nDimension == 3 && t->size[2] == 2;
}

static int nn_(checkKvInput)(THLongTensor* key, THTensor* val) {
  return THLongTensor_nDimension(key) == 2 &&
    THTensor_(nDimension)(val) == 2 &&
    THTensor_(size)(val, 0) == THLongTensor_size(key, 0) &&
    THTensor_(size)(val, 1) == THLongTensor_size(key, 1);
}

static int nn_(checkSize2D)(THTensor* t, long size0, long size1) {
  return t->nDimension == 2 && t->size[0] == size0 && t->size[1] == size1;
}

static int nn_(checkSize1D)(THTensor* t, long size0) {
  return t->nDimension == 1 && t->size[0] == size0;
}

static void nn_(set1d)(THTensor *t, long x0, real value) {
  THStorage_(set)(t->storage, t->storageOffset + x0*t->stride[0], value);
}
static real nn_(get3d)(const THTensor *t, long x0, long x1, long x2) {
  return THStorage_(get)(t->storage, t->storageOffset +
                         x0*t->stride[0] + x1*t->stride[1] + x2*t->stride[2]);
}
static real nn_(get2d)(const THTensor *t, long x0, long x1) {
  return THStorage_(get)(t->storage, t->storageOffset +
                         x0*t->stride[0] + x1*t->stride[1]);
}
static long nn_(get2dL)(const THLongTensor *t, long x0, long x1) {
  return THLongStorage_get(t->storage, t->storageOffset +
                           x0*t->stride[0] + x1*t->stride[1]);
}

static int nn_(SparseLinear_updateOutput)(lua_State* L) {
  long h, i;
  THTensor* input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor* weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor* bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor* output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  long outDim = THTensor_(size)(weight, 0);
  long inDim = THTensor_(size)(weight, 1);

  luaL_argcheck(L, nn_(checkSize1D)(bias, outDim), 1, "bias size wrong");
  luaL_argcheck(L, nn_(checkInput)(input), 2,
                "input size must be batchsize x nnz x 2");
  luaL_argcheck(L, THTensor_(isContiguous)(output), 1,
                "output must be contiguous");

  long batchSize = THTensor_(size)(input, 0);
  long nnz = THTensor_(size)(input, 1);
  THTensor_(resize2d)(output, batchSize, outDim);

  // output = weight * input + bias
  THTensor_(zero)(output);
#pragma omp parallel for private(h, i) schedule(static) if (   \
  batchSize > 1 && batchSize * nnz * outDim > 10000)
  for (h = 0; h < batchSize; h++) {
    for (i = 0; i < nnz; i++) {
      real val = nn_(get3d)(input, h, i, 1);
      if (val == 0) {
        continue;
      }

      long offset = (long)(nn_(get3d)(input, h, i, 0)) - 1;
      if (offset >= 0 && offset < inDim) {
        THBlas_(axpy)(outDim,
                      val,
                      COL_PTR2(weight, offset), weight->stride[0],
                      ROW_PTR2(output, h), output->stride[1]);
      } else {
        luaL_error(
          L,
          "index out of bound. updateOutput: %d not between 1 and %d",
          offset + 1,
          inDim);
      }
    }
  }

  THTensor* output_row = THTensor_(new)();
  for (h = 0; h < batchSize; h++) {
    THTensor_(select)(output_row, output, 0, h);
    THTensor_(cadd)(output_row, bias, 1.0, output_row);
  }
  THTensor_(free)(output_row);

  if (batchSize == 1) {
    THTensor_(resize1d)(output, outDim);
  }

  lua_getfield(L, 1, "output");
  return 1;
}

static int nn_(SparseLinear_updateOutput2)(lua_State* L) {
  long h, i;
  THLongTensor* inputKey = luaT_checkudata(L, 2, "torch.LongTensor");
  THTensor* inputVal = luaT_checkudata(L, 3, torch_Tensor);
  THTensor* weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor* bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor* output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  long outDim = THTensor_(size)(weight, 0);
  long inDim = THTensor_(size)(weight, 1);

  luaL_argcheck(L, nn_(checkSize1D)(bias, outDim), 1, "bias size wrong");
  luaL_argcheck(L, nn_(checkKvInput)(inputKey, inputVal), 3, "input wrong");
  luaL_argcheck(L, THTensor_(isContiguous)(output), 1,
                "output must be contiguous");

  long batchSize = THLongTensor_size(inputKey, 0);
  long nnz = THLongTensor_size(inputKey, 1);
  THTensor_(resize2d)(output, batchSize, outDim);

  // output = weight * input + bias
  THTensor_(zero)(output);
#pragma omp parallel for private(h, i) schedule(static) if (   \
  batchSize > 1 && batchSize * nnz * outDim > 10000)
  for (h = 0; h < batchSize; ++h) {
    for (i = 0; i < nnz; ++i) {
      real val = nn_(get2d)(inputVal, h, i);
      if (val == 0) {
        continue;
      }

      long offset = nn_(get2dL)(inputKey, h, i) - 1;
      if (offset >= 0 && offset < inDim) {
        THBlas_(axpy)(outDim,
                      val,
                      COL_PTR2(weight, offset), weight->stride[0],
                      ROW_PTR2(output, h), output->stride[1]);
      } else {
        luaL_error(
          L, "wrong index. updateOutput2: %d vs %d", offset + 1, inDim);
      }
    }
  }

  THTensor* output_row = THTensor_(new)();
  for (h = 0; h < batchSize; ++h) {
    THTensor_(select)(output_row, output, 0, h);
    THTensor_(cadd)(output_row, bias, 1.0, output_row);
  }
  THTensor_(free)(output_row);

  if (batchSize == 1) {
    THTensor_(resize1d)(output, outDim);
  }

  lua_getfield(L, 1, "output");
  return 1;
}

static int nn_(SparseLinear_accGradParameters)(lua_State* L) {
  long h, i;
  THTensor* input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor* gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real scale = luaL_optnumber(L, 4, 1);
  THTensor* weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor* gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);
  THTensor* gradWeight = luaT_getfieldcheckudata(
    L, 1, "gradWeight", torch_Tensor);
  real weightDecay = luaT_getfieldchecknumber(L, 1, "weightDecay");

  long outDim = THTensor_(size)(weight, 0);
  long inDim = THTensor_(size)(weight, 1);

  luaL_argcheck(L, nn_(checkSize2D)(gradWeight, outDim, inDim), 1,
                "gradWeight size wrong");
  luaL_argcheck(L, nn_(checkSize1D)(gradBias, outDim), 1,
                "gradBias size wrong");
  luaL_argcheck(L, nn_(checkInput)(input), 2,
                "input must be a batchsize x nnz x 2");
  luaL_argcheck(L, THTensor_(isContiguous)(gradOutput), 1,
                "output must be contiguous");

  long batchSize = THTensor_(size)(input, 0);
  long nnz = THTensor_(size)(input, 1);
  THTensor_(resize2d)(gradOutput, batchSize, outDim);

  // gradWeight += gradOutput * input
#pragma omp parallel for private(h, i) schedule(static) if (\
  batchSize * nnz * outDim > 10000)
  for (i = 0; i < nnz; i++) {
    for (h = 0; h < batchSize; h++) {
      real val = scale * nn_(get3d)(input, h, i, 1);
      if (val == 0) {
        continue;
      }

      long offset = (long)(nn_(get3d)(input, h, i, 0)) - 1;
      if (offset >= 0 && offset < inDim) {
        THBlas_(axpy)(outDim,
                      val,
                      ROW_PTR2(gradOutput, h), gradOutput->stride[1],
                      COL_PTR2(gradWeight, offset), gradWeight->stride[0]);
      } else {
        luaL_error(
          L,
          "index out of bound. accGradParameters: %d not between 1 and %d",
          offset + 1,
          inDim);
      }
    }
  }

  // gradBias += gradOutput
  THTensor* gradOutput_row = THTensor_(new)();
  for (h = 0; h < batchSize; h++) {
    THTensor_(select)(gradOutput_row, gradOutput, 0, h);
    THTensor_(cadd)(gradBias, gradBias, scale, gradOutput_row);
  }
  THTensor_(free)(gradOutput_row);

  if (weightDecay != 0) {
    THTensor_(cadd)(gradWeight, gradWeight, weightDecay, weight);
  }

  if (batchSize == 1) {
    THTensor_(resize1d)(gradOutput, outDim);
  }
  return 0;
}

static int nn_(SparseLinear_accGradParameters2)(lua_State* L) {
  long h, i;
  THLongTensor* inputKey = luaT_checkudata(L, 2, "torch.LongTensor");
  THTensor* inputVal = luaT_checkudata(L, 3, torch_Tensor);
  THTensor* gradOutput = luaT_checkudata(L, 4, torch_Tensor);
  real scale = luaL_optnumber(L, 5, 1);
  THTensor* weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor* gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);
  THTensor* gradWeight = luaT_getfieldcheckudata(
    L, 1, "gradWeight", torch_Tensor);
  real weightDecay = luaT_getfieldchecknumber(L, 1, "weightDecay");

  long outDim = THTensor_(size)(weight, 0);
  long inDim = THTensor_(size)(weight, 1);

  luaL_argcheck(L, nn_(checkSize2D)(gradWeight, outDim, inDim), 1,
                "gradWeight size wrong");
  luaL_argcheck(L, nn_(checkSize1D)(gradBias, outDim), 1,
                "gradBias size wrong");
  luaL_argcheck(L, nn_(checkKvInput)(inputKey, inputVal), 3, "input wrong");
  luaL_argcheck(L, THTensor_(isContiguous)(gradOutput), 1,
                "output must be contiguous");

  // unify dimensions for batch and non-batch input
  long batchSize = THLongTensor_size(inputKey, 0);
  long nnz = THLongTensor_size(inputKey, 1);
  THTensor_(resize2d)(gradOutput, batchSize, outDim);

  // gradWeight += gradOutput * input
#pragma omp parallel for private(h, i) schedule(static) if (\
  batchSize * nnz * outDim > 10000)
  for (i = 0; i < nnz; ++i) {
    for (h = 0; h < batchSize; ++h) {
      real val = scale * nn_(get2d)(inputVal, h, i);
      if (val == 0) {
        continue;
      }

      long offset = nn_(get2dL)(inputKey, h, i) - 1;
      if (offset >= 0 && offset < inDim) {
        THBlas_(axpy)(outDim,
                      val,
                      ROW_PTR2(gradOutput, h), gradOutput->stride[1],
                      COL_PTR2(gradWeight, offset), gradWeight->stride[0]);
      } else {
        luaL_error(
          L, "wrong index. accGradParameters: %d vs %d", offset + 1, inDim);
      }
    }
  }

  // gradBias += gradOutput
  THTensor* gradOutput_row = THTensor_(new)();
  for (h = 0; h < batchSize; h++) {
    THTensor_(select)(gradOutput_row, gradOutput, 0, h);
    THTensor_(cadd)(gradBias, gradBias, scale, gradOutput_row);
  }
  THTensor_(free)(gradOutput_row);

  if (weightDecay != 0) {
    THTensor_(cadd)(gradWeight, gradWeight, weightDecay, weight);
  }

  if (batchSize == 1) {
    THTensor_(resize1d)(gradOutput, outDim);
  }
  return 0;
}

int nn_(SparseLinear_updateParameters)(lua_State* L) {
  long h, i;
  real learningRate = luaL_checknumber(L, 2);
  THTensor* weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor* bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor* gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);
  THTensor* gradWeight = luaT_getfieldcheckudata(
    L, 1, "gradWeight", torch_Tensor);
  THTensor* lastInput = luaT_getfieldcheckudata(
    L, 1, "lastInput", torch_Tensor);

  long outDim = THTensor_(size)(weight, 0);
  long inDim = THTensor_(size)(weight, 1);

  luaL_argcheck(L, nn_(checkSize2D)(gradWeight, outDim, inDim), 1,
                "gradWeight size wrong");
  luaL_argcheck(L, nn_(checkSize1D)(bias, outDim), 1, "bias size wrong");
  luaL_argcheck(L, nn_(checkSize1D)(gradBias, outDim), 1,
                                    "gradBias size wrong");
  luaL_argcheck(L, nn_(checkInput)(lastInput), 1,
                "input size must be batchsize x nnz x 2");

  long batchSize = THTensor_(size)(lastInput, 0);
  long nnz = THTensor_(size)(lastInput, 1);

  // collect unique offsets of non-0 val in input
  THTensor* offsets = THTensor_(newWithSize1d)(batchSize * nnz);
  long cnt = 0;
  for (h = 0; h < batchSize; h++) {
    for (i = 0; i < nnz; i++) {
      real val = nn_(get3d)(lastInput, h, i, 1);
      if (val == 0 ) {
        continue;
      }
      long offset = (long)(nn_(get3d)(lastInput, h, i, 0)) - 1;
      if (offset >= 0 && offset < inDim) {
        nn_(set1d)(offsets, cnt++, offset);
      } else {
        luaL_error(
          L,
          "index out of bound. updateParameters: %d not between 1 and %d",
          offset + 1,
          inDim);
      }
    }
  }
  THTensor_(resize1d)(offsets, cnt);

  THTensor* uniqueOffsets = THTensor_(new)();
  THLongTensor* ri = THLongTensor_new();
  THTensor_(sort)(uniqueOffsets, ri, offsets, 0, 0);
  THLongTensor_free(ri);
  THTensor_(free)(offsets);

  cnt = 1;
  real* uniqueOffsets_p = THTensor_(data)(uniqueOffsets);
  for (i = 1; i < THTensor_(size)(uniqueOffsets, 0); i++) {
    if (uniqueOffsets_p[i] != uniqueOffsets_p[i - 1]) {
      uniqueOffsets_p[cnt++] = uniqueOffsets_p[i];
    }
  }
  THTensor_(resize1d)(uniqueOffsets, cnt);

  // weight += -learningRate * gradWeight
  THTensor_(cadd)(bias, bias, -learningRate, gradBias);
#pragma omp parallel for private(i) schedule(static) if (cnt * outDim > 10000)
  for (i = 0; i < cnt; i++) {
    long offset = (long)uniqueOffsets_p[i];
    THBlas_(axpy)(outDim,
                  -learningRate,
                  COL_PTR2(gradWeight, offset), gradWeight->stride[0],
                  COL_PTR2(weight, offset), weight->stride[0]);
  }

  THTensor_(free)(uniqueOffsets);

  return 0;
}

int nn_(SparseLinear_updateParameters2)(lua_State* L) {
  long h, i;
  real learningRate = luaL_checknumber(L, 2);
  THTensor* weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor* bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor* gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);
  THTensor* gradWeight = luaT_getfieldcheckudata(
    L, 1, "gradWeight", torch_Tensor);
  THLongTensor* lastInputKey = luaT_getfieldcheckudata(
    L, 1, "lastInputKey", "torch.LongTensor");
  THTensor* lastInputVal = luaT_getfieldcheckudata(
    L, 1, "lastInputVal", torch_Tensor);

  long outDim = THTensor_(size)(weight, 0);
  long inDim = THTensor_(size)(weight, 1);

  luaL_argcheck(L, nn_(checkSize2D)(gradWeight, outDim, inDim), 1,
                "gradWeight size wrong");
  luaL_argcheck(L, nn_(checkSize1D)(bias, outDim), 1, "bias size wrong");
  luaL_argcheck(L, nn_(checkSize1D)(gradBias, outDim), 1,
                                    "gradBias size wrong");
  luaL_argcheck(
    L, nn_(checkKvInput)(lastInputKey, lastInputVal), 1, "input wrong");

  long batchSize = THLongTensor_size(lastInputKey, 0);
  long nnz = THLongTensor_size(lastInputKey, 1);

  // collect unique offsets of non-0 val in input
  THLongTensor* offsets = THLongTensor_newWithSize1d(batchSize * nnz);
  long* offsets_p = THLongTensor_data(offsets);
  long cnt = 0;
  for (h = 0; h < batchSize; ++h) {
    for (i = 0; i < nnz; ++i) {
      real val = nn_(get2d)(lastInputVal, h, i);
      if (val == 0 ) {
        continue;
      }
      long offset = nn_(get2dL)(lastInputKey, h, i) - 1;
      if (offset >= 0 && offset < inDim) {
        offsets_p[cnt++] = offset;
      } else {
        luaL_error(
          L, "index wrong. updateParameters: %d vs %d", offset + 1, inDim);
      }
    }
  }
  THLongTensor_resize1d(offsets, cnt);

  THLongTensor* uniqueOffsets = THLongTensor_new();
  THLongTensor* ri = THLongTensor_new();
  THLongTensor_sort(uniqueOffsets, ri, offsets, 0, 0);
  THLongTensor_free(ri);
  THLongTensor_free(offsets);

  cnt = 1;
  long* uniqueOffsets_p = THLongTensor_data(uniqueOffsets);
  for (i = 1; i < THLongTensor_size(uniqueOffsets, 0); ++i) {
    if (uniqueOffsets_p[i] != uniqueOffsets_p[i - 1]) {
      uniqueOffsets_p[cnt++] = uniqueOffsets_p[i];
    }
  }
  THLongTensor_resize1d(uniqueOffsets, cnt);

  // weight += -learningRate * gradWeight
  THTensor_(cadd)(bias, bias, -learningRate, gradBias);
#pragma omp parallel for private(i) schedule(static) if (cnt * outDim > 10000)
  for (i = 0; i < cnt; ++i) {
    long offset = uniqueOffsets_p[i];
    THBlas_(axpy)(outDim,
                  -learningRate,
                  COL_PTR2(gradWeight, offset), gradWeight->stride[0],
                  COL_PTR2(weight, offset), weight->stride[0]);
  }

  THLongTensor_free(uniqueOffsets);

  return 0;
}

int nn_(SparseLinear_zeroGradParameters)(lua_State* L) {
  long h, i, j;
  THTensor* gradBias = luaT_getfieldcheckudata(
    L, 1, "gradBias", torch_Tensor);
  THTensor* gradWeight = luaT_getfieldcheckudata(
    L, 1, "gradWeight", torch_Tensor);
  THTensor* lastInput = luaT_getfieldcheckudata(
    L, 1, "lastInput", torch_Tensor);

  long outDim = THTensor_(size)(gradWeight, 0);
  long inDim = THTensor_(size)(gradWeight, 1);

  luaL_argcheck(
    L, nn_(checkSize1D)(gradBias, outDim), 1, "gradBias size wrong");
  luaL_argcheck(L, nn_(checkInput)(lastInput), 1,
                "input size must be batchsize x nnz x 2");

  THTensor_(zero)(gradBias);

  long batchSize = THTensor_(size)(lastInput, 0);
  long nnz = THTensor_(size)(lastInput, 1);

#pragma omp parallel for private(h, i, j) schedule(static) if (   \
  batchSize > 1 && batchSize * nnz * outDim > 10000)
  for (h = 0; h < batchSize; h++) {
    for (i = 0; i < nnz; i++) {
      if (nn_(get3d)(lastInput, h, i, 1) == 0 ) {
        continue;
      }

      long offset = (long)(nn_(get3d)(lastInput, h, i, 0)) - 1;
      if (offset >= 0 && offset < inDim) {
        real* pGradWeight = COL_PTR2(gradWeight, offset);
        if (gradWeight->stride[0] == 1) {
          THVector_(fill)(pGradWeight, 0, outDim);
        } else {
          long stride = gradWeight->stride[0];
          for (j = 0; j < outDim; ++j) {
            pGradWeight[j * stride] = 0;
          }
        }
      } else {
        luaL_error(
          L,
          "index out of bound. zeroGradParameters: %d not between 1 and %d",
          offset + 1,
          inDim);
      }
    }
  }

  return 0;
}

int nn_(SparseLinear_zeroGradParameters2)(lua_State* L) {
  long h, i, j;
  THTensor* gradBias = luaT_getfieldcheckudata(
    L, 1, "gradBias", torch_Tensor);
  THTensor* gradWeight = luaT_getfieldcheckudata(
    L, 1, "gradWeight", torch_Tensor);
  THLongTensor* lastInputKey = luaT_getfieldcheckudata(
    L, 1, "lastInputKey", "torch.LongTensor");
  THTensor* lastInputVal = luaT_getfieldcheckudata(
    L, 1, "lastInputVal", torch_Tensor);

  long outDim = THTensor_(size)(gradWeight, 0);
  long inDim = THTensor_(size)(gradWeight, 1);

  luaL_argcheck(
    L, nn_(checkKvInput)(lastInputKey, lastInputVal), 1, "input wrong");
  luaL_argcheck(
    L, nn_(checkSize1D)(gradBias, outDim), 1, "gradBias size wrong");

  THTensor_(zero)(gradBias);

  long batchSize = THLongTensor_size(lastInputKey, 0);
  long nnz = THLongTensor_size(lastInputKey, 1);

#pragma omp parallel for private(h, i, j) schedule(static) if (   \
  batchSize > 1 && batchSize * nnz * outDim > 10000)
  for (h = 0; h < batchSize; ++h) {
    for (i = 0; i < nnz; ++i) {
      if (nn_(get2d)(lastInputVal, h, i) == 0 ) {
        continue;
      }

      long offset = nn_(get2dL)(lastInputKey, h, i) - 1;
      if (offset >= 0 && offset < inDim) {
        real* pGradWeight = COL_PTR2(gradWeight, offset);
        if (gradWeight->stride[0] == 1) {
          THVector_(fill)(pGradWeight, 0, outDim);
        } else {
          long stride = gradWeight->stride[0];
          for (j = 0; j < outDim; ++j) {
            pGradWeight[j * stride] = 0;
          }
        }
      } else {
        luaL_error(
          L, "index wrong. zeroGradParameters: %d vs %d", offset + 1, inDim);
      }
    }
  }

  return 0;
}

static int nn_(SparseLinear_updateGradInput)(lua_State* L) {
  THTensor* weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor* gradInput =
      luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  THTensor* input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor* gradOutput = luaT_checkudata(L, 3, torch_Tensor);

  long h, i;
  long outDim = THTensor_(size)(weight, 0);
  long inDim = THTensor_(size)(weight, 1);

  luaL_argcheck(L, nn_(checkInput)(input), 2,
                "input must be a batchsize x nnz x 2 or nnz x 2 tensor");
  luaL_argcheck(L, THTensor_(isContiguous)(gradInput), 1,
                "gradInput must be contiguous");
  luaL_argcheck(L, THTensor_(isContiguous)(gradOutput), 1,
                "gradOutput must be contiguous");

  long batchSize = THTensor_(size)(input, 0);
  long nnz = THTensor_(size)(input, 1);
  THTensor_(resize2d)(gradOutput, batchSize, outDim);
  THTensor_(resize3d)(gradInput, batchSize, nnz, 2);

#pragma omp parallel for private(h, i) schedule(static) if (\
  batchSize > 1 && batchSize * nnz * outDim > 10000)
  for (h = 0; h < batchSize; h++) {
    for (i = 0; i < nnz; ++i) {
      long offset = (long)(THTensor_(get3d)(input, h, i, 0)) - 1;
      THTensor_(set3d)(gradInput, h, i, 0, offset + 1);

      if (offset >= 0 && offset < inDim) {
        real val = THBlas_(dot)(
            outDim,
            ROW_PTR2(gradOutput, h), gradOutput->stride[1],
            COL_PTR2(weight, offset), weight->stride[0]);
        THTensor_(set3d)(gradInput, h, i, 1, val);
      } else {
        luaL_error(
          L,
          "index out of bound. updateGradInput: %d not between 1 and %d",
          offset + 1,
          inDim);
      }
    }
  }

  if (batchSize == 1) {
    THTensor_(resize1d)(gradOutput, outDim);
    THTensor_(resize2d)(gradInput, nnz, 2);
  }
  return 0;
}

static const struct luaL_Reg nn_(SparseLinear__)[] = {
    {"SparseLinear_updateOutput", nn_(SparseLinear_updateOutput)},
    {"SparseLinear_accGradParameters", nn_(SparseLinear_accGradParameters)},
    {"SparseLinear_updateParameters", nn_(SparseLinear_updateParameters)},
    {"SparseLinear_zeroGradParameters", nn_(SparseLinear_zeroGradParameters)},
    {"SparseLinear_updateGradInput", nn_(SparseLinear_updateGradInput)},
    {"SparseLinear_updateOutput2", nn_(SparseLinear_updateOutput2)},
    {"SparseLinear_accGradParameters2", nn_(SparseLinear_accGradParameters2)},
    {"SparseLinear_updateParameters2", nn_(SparseLinear_updateParameters2)},
    {"SparseLinear_zeroGradParameters2", nn_(SparseLinear_zeroGradParameters2)},
    {NULL, NULL}};

void nn_(SparseLinear_init)(lua_State* L) {
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SparseLinear__), "nn");
  lua_pop(L, 1);
}

#undef ROW_PTR2
#undef COL_PTR2

#endif
