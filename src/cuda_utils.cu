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
void kMulMtx(T * ptr, T val)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  ptr[i] *= val;
}

void fillMtxf(float * ptr, int size, float val)
{
  kFillMtx<float> <<<1, size>>> (ptr, val);
}

void addMtxf(float * ptr, int size, float val)
{
  kAddMtx<float> <<<1, size>>> (ptr, val);
}

void mulMtxf(float * ptr, int size, float val)
{
  kMulMtx<float> <<<1, size>>> (ptr, val);
}
