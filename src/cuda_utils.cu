#include <iostream>
#include "cuda_utils.hpp"

using namespace std;

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

void mulMtxf(float * ptr, int size, float val)
{
  kMulMtx<float> <<<1, size>>> (ptr, val);
}
