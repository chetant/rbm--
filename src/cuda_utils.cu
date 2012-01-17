#include <iostream>
#include "cuda_utils.hpp"

using namespace std;

CUDADevice CUDASystem::currDevice;

#define sigmoidd(x) (1 / (1 + __expf(-x)))

template<typename T>
__global__ 
void kFillMtx(T * ptr, T val)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  ptr[i] = val;
}

template<typename T>
__global__ 
void kAddMtx(T * ptr, T val)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  ptr[i] += val;
}

template<typename T>
__global__ 
void kAddMtxMtx(T * toptr, const T* fromptr)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  toptr[i] += fromptr[i];
}

template<typename T>
__global__ 
void kSubMtxMtx(T * toptr, const T* fromptr)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  toptr[i] -= fromptr[i];
}

template<typename T>
__global__ 
void kGTMtxMtx(T * toptr, const T* fromptr)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  toptr[i] = toptr[i] > fromptr[i] ? 1 : 0;
}

template<typename T>
__global__ 
void kMulMtx(T * ptr, T val)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  ptr[i] *= val;
}

__global__ 
void setupRandSt(curandState * state, int seed)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, i, 0, &state[i]);
}

__global__ 
void sampleRand(float * data, curandState * state)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  data[i] = curand_uniform(&state[i]);
}

__global__ 
void kSampleVis(float * v, float * vs, curandState * state, int numSamples)
{
  const int i = threadIdx.x;
  const int j = threadIdx.x * numSamples + threadIdx.y;
  vs[j] = signbit(curand_uniform(&state[j]) - sigmoidd(v[i]));
}

void sampleVis(float * v, float * vs, curandState * randStates, int numSamples, int numVisible)
{
  dim3 dimBlock(numVisible, numSamples);
  kSampleVis<<<1, dimBlock>>>(v, vs, randStates, numSamples);
}

__global__ 
void kSampleHid(float * hs, curandState * state, int numSamples)
{
  const int i = threadIdx.x * numSamples + threadIdx.y;
  hs[i] = signbit(curand_uniform(&state[i]) - hs[i]);
}

void sampleHid(float * hs, curandState * randStates, int numSamples, int numHidden)
{
  dim3 dimBlock(numHidden, numSamples);
  kSampleHid<<<1, dimBlock>>>(hs, randStates, numSamples);
}

void setupRandStates(curandState * state, int size, int seed)
{
  setupRandSt<<<1, size>>>(state, seed);
}

void fillRand(float * data, int size, curandState * state)
{
  sampleRand<<<1, size>>>(data, state);
}

void fillMtxf(float * ptr, int size, float val)
{
  kFillMtx<float> <<<1, size>>> (ptr, val);
}

void addMtxf(float * ptr, int size, float val)
{
  kAddMtx<float> <<<1, size>>> (ptr, val);
}

void addMtxMtxf(float * toptr, const float * fromptr, int size)
{
  kAddMtxMtx<float> <<<1, size>>> (toptr, fromptr);
}

void subMtxMtxf(float * toptr, const float * fromptr, int size)
{
  kSubMtxMtx<float> <<<1, size>>> (toptr, fromptr);
}

template<typename T>
__global__ 
void kSubMtxRVec(T * toptr, const T* fromptr, int numCols)
{
  const int i = threadIdx.x * numCols + threadIdx.y;
  const int j = threadIdx.x;
  toptr[i] -= fromptr[j];
}

void subMtxRVecf(float * toptr, const float * fromptr, int numRows, int numCols)
{
  dim3 dimBlocks(numRows, numCols);
  kSubMtxRVec<float> <<<1, dimBlocks>>> (toptr, fromptr, numCols);
}

template<typename T>
__global__ 
void kAddMtxRVec(T * toptr, const T* fromptr, int numCols)
{
  const int i = threadIdx.x * numCols + threadIdx.y;
  const int j = threadIdx.x;
  toptr[i] += fromptr[j];
}

void addMtxRVecf(float * toptr, const float * fromptr, int numRows, int numCols)
{
  dim3 dimBlocks(numRows, numCols);
  kAddMtxRVec<float> <<<1, dimBlocks>>> (toptr, fromptr, numCols);
}

void gtMtxMtxf(float * toptr, const float * fromptr, int size)
{
  kGTMtxMtx<float> <<<1, size>>> (toptr, fromptr);
}

void mulMtxf(float * ptr, int size, float val)
{
  kMulMtx<float> <<<1, size>>> (ptr, val);
}

template<typename T>
__global__ 
void kMulMtxMtx(T * toptr, const T* fromptr)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  toptr[i] += fromptr[i];
}

void mulMtxMtxf(float * toptr, const float * fromptr, int size)
{
  kMulMtxMtx<float> <<<1, size>>> (toptr, fromptr);
}
