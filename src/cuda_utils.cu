#include <iostream>
#include "cuda_utils.hpp"

using namespace std;

CUDADevice CUDASystem::currDevice;

#define sigmoidd(x) (1 / (1 + __expf(-x)))
#define split(x) const int maxt = CUDASystem::currDevice.maxThreadsPerBlockPerSide(); \
                 int oX = (maxt % x || maxt > x) ? 1 : 0; \
                 dim3 dimGrid(x/maxt + oX); \
                 dim3 dimBlock(maxt);
#define splitGrid(x, y) const int maxt = CUDASystem::currDevice.maxThreadsPerBlockPerSide(); \
                        int oX = (maxt % x || maxt > x) ? 1 : 0; \
                        int oY = (maxt % y || maxt > y) ? 1 : 0; \
                        dim3 dimGrid(x/maxt + oX, y/maxt + oY); \
                        dim3 dimBlock(maxt, maxt);

template<typename T>
__global__ 
void kFillMtx(T * ptr, T val, const int size)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < size)
    ptr[i] = val;
}

template<typename T>
__global__ 
void kAddMtx(T * ptr, T val, const int size)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < size)
    ptr[i] += val;
}

template<typename T>
__global__ 
void kAddMtxMtx(T * toptr, const T* fromptr, const int size)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < size)
    toptr[i] += fromptr[i];
}

template<typename T>
__global__ 
void kSubMtxMtx(T * toptr, const T* fromptr, const int size)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < size)
    toptr[i] -= fromptr[i];
}

template<typename T>
__global__ 
void kGTMtxMtx(T * toptr, const T* fromptr, const int size)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < size)
    toptr[i] = toptr[i] > fromptr[i] ? 1 : 0;
}

template<typename T>
__global__ 
void kMulMtx(T * ptr, T val, const int size)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < size)
    ptr[i] *= val;
}

__global__ 
void setupRandSt(curandState * state, const int seed, const int size)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < size)
    curand_init(seed, i, 0, &state[i]);
}

__global__ 
void sampleRandU(float * data, curandState * state, const int size)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < size)
    data[i] = curand_uniform(&state[i]);
}

__global__ 
void sampleRandN(float * data, curandState * state, const int size, const float mean, const float sd)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < size)
    data[i] = curand_normal(&state[i]) * sd + mean;
}

// Sample from visible vector to vs
__global__ 
void kSampleVis(float * v, float * vs, curandState * state, const int numSamples, const int numVisible)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  // const int j = (blockIdx.x * blockDim.x + threadIdx.x) * numSamples + (blockIdx.y * blockDim.y + threadIdx.y);
  if(i < numVisible && j < numSamples)
  {
    const int k = i*numSamples + j;
    vs[k] = signbit(curand_uniform(&state[k]) - sigmoidd(v[i]));
  }
}

void sampleVis(float * v, float * vs, curandState * randStates, const int numSamples, const int numVisible)
{
  splitGrid(numVisible, numSamples);
  kSampleVis<<<dimGrid, dimBlock>>>(v, vs, randStates, numSamples, numVisible);
}

// Sample from phs to hs
__global__ 
void kSampleHid(float * hs, curandState * state, const int numSamples, const int numHidden)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  // const int i = (blockIdx.x * blockDim.x + threadIdx.x) * numSamples + (blockIdx.y * blockDim.y + threadIdx.y);
  if(i < numHidden && j < numSamples)
  {
    const int k = i*numSamples + j;
    hs[k] = signbit(curand_uniform(&state[k]) - hs[k]);
  }
}

void sampleHid(float * hs, curandState * randStates, const int numSamples, const int numHidden)
{
  splitGrid(numHidden, numSamples);
  kSampleHid<<<dimGrid, dimBlock>>>(hs, randStates, numSamples, numHidden);
}

// setup initial rand states
void setupRandStates(curandState * state, int size, const int seed)
{
  split(size);
  setupRandSt<<<dimGrid, dimBlock>>>(state, seed, size);
}

void fillRandU(float * data, int size, curandState * state)
{
  split(size);
  sampleRandU<<<dimGrid, dimBlock>>>(data, state, size);
}

void fillRandN(float * data, int size, curandState * state, float mean, float sd)
{
  split(size);
  sampleRandN<<<dimGrid, dimBlock>>>(data, state, size, mean, sd);
}

void fillMtxf(float * ptr, int size, float val)
{
  split(size);
  kFillMtx<float> <<<dimGrid, dimBlock>>> (ptr, val, size);
}

void addMtxf(float * ptr, int size, float val)
{
  split(size);
  kAddMtx<float> <<<dimGrid, dimBlock>>> (ptr, val, size);
}

void addMtxMtxf(float * toptr, const float * fromptr, int size)
{
  split(size);
  kAddMtxMtx<float> <<<dimGrid, dimBlock>>> (toptr, fromptr, size);
}

void subMtxMtxf(float * toptr, const float * fromptr, int size)
{
  split(size);
  kSubMtxMtx<float> <<<dimGrid, dimBlock>>> (toptr, fromptr, size);
}

template<typename T>
__global__ 
void kSubMtxRVec(T * toptr, const T* fromptr, int numRows, int numCols)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i < numCols && j < numRows)
  {
    const int k = i*numCols + j;
    toptr[k] -= fromptr[i];
  }
}

void subMtxRVecf(float * toptr, const float * fromptr, int numRows, int numCols)
{
  dim3 dimBlocks(numRows, numCols);
  kSubMtxRVec<float> <<<1, dimBlocks>>> (toptr, fromptr, numRows, numCols);
}

template<typename T>
__global__ 
void kAddMtxRVec(T * toptr, const T* fromptr, int numRows, int numCols)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i < numCols && j < numRows)
  {
    const int k = i*numCols + j;
    toptr[k] += fromptr[i];
  }
}

void addMtxRVecf(float * toptr, const float * fromptr, int numRows, int numCols)
{
  dim3 dimBlocks(numRows, numCols);
  kAddMtxRVec<float> <<<1, dimBlocks>>> (toptr, fromptr, numRows, numCols);
}

void gtMtxMtxf(float * toptr, const float * fromptr, int size)
{
  kGTMtxMtx<float> <<<1, size>>> (toptr, fromptr, size);
}

void mulMtxf(float * ptr, int size, float val)
{
  kMulMtx<float> <<<1, size>>> (ptr, val, size);
}

template<typename T>
__global__ 
void kMulMtxMtx(T * toptr, const T* fromptr, int size)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < size)
    toptr[i] *= fromptr[i];
}

void mulMtxMtxf(float * toptr, const float * fromptr, int size)
{
  kMulMtxMtx<float> <<<1, size>>> (toptr, fromptr, size);
}
