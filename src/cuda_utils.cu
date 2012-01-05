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
void kMulMtx(T * ptr, T val)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  ptr[i] *= val;
}

void fillMtxf(float * ptr, int size, float val)
{
  kFillMtx<float> <<<1, 6>>> (ptr, val);
}

void mulMtxf(float * ptr, int size, float val)
{
  kMulMtx<float> <<<1, 6>>> (ptr, val);
}

// __global__ 
// void kFillMtxf(float * ptr, float val)
// {
//   const int i = blockIdx.x * blockDim.x + threadIdx.x;
//   ptr[i] = val;
// }

// void test()
// {
//   const int N = 6;
//   float * devPtr;
//   if(cudaSuccess != cudaMalloc(&devPtr, N*sizeof(float)))
//   {
//     std::cerr << "Cannot alloc device memory of size" << N << std::endl;
//     return;
//   }

//   float * hostPtr = new float[N];
//   for(int i = 0; i < N; ++i)
//     hostPtr[i] = i;

//   kFillMtxf<<< 1, N >>>(devPtr, 42.5);

//   cudaMemcpy(hostPtr, devPtr, N*sizeof(float), cudaMemcpyDeviceToHost);

//   for(int i = 0; i < N; ++i)
//     cout << hostPtr[i] << '\t';
//   cout << endl;

//   cudaFree(devPtr);
//   delete hostPtr;
// }
