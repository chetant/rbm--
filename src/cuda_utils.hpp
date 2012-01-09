#ifndef _CUDA_UTILS_H_
#define _CUDA_UTILS_H_

#include <iostream>
#include <string>

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

using std::cerr;
using std::endl;
using std::string;

void setupRandStates(curandState * state, int size, int seed);
void fillRand(float * data, int size, curandState * state);

void fillMtxf(float * ptr, int size, float val);
void mulMtxf(float * ptr, int size, float val);
void addMtxf(float * ptr, int size, float val);

void addMtxMtxf(float * toptr, const float * fromptr, int size);

// CUDA init
class CUDADevice
{
  int devId;
  cudaDeviceProp prop;
public:
  CUDADevice(int i = 0) : devId(-1)
  {
    if(cudaSuccess != cudaGetDeviceProperties(&prop, i))
    {
      cerr << "ERROR: Cannot query CUDA device properties!!" << endl;
    }
    devId = i;
  }

  string name() { return string(prop.name); }

  int multiProcs() { if(devId < 0) return -1; else return prop.multiProcessorCount;}
  int warpSize() { if(devId < 0) return -1; else return prop.warpSize;}
  double computeCaps() { if(devId < 0) return -1; else return prop.major + prop.minor/10.0;}
  bool unifiedAddx() { if(devId < 0) return false; else return prop.unifiedAddressing;}
  bool canMapHostMem() { if(devId < 0) return false; else return prop.canMapHostMemory;}
  int maxThreadsPerMP() { if(devId < 0) return false; else return prop.maxThreadsPerMultiProcessor; } 
  int maxThreads() { if(devId < 0) return false; else return prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount; }
};

class CUDASystem
{
public:
  static int getNumDevices()
  {
    int n = 0;
    cudaError_t err = cudaGetDeviceCount(&n);
    if(cudaErrorNoDevice == err)
    {
      cerr << "ERROR: No CUDA compatible device found!!" << endl;
    }
    else if(cudaErrorInsufficientDriver == err)
    {
      cerr << "ERROR: Driver does not have CUDA support(Hw may support CUDA)!!" << endl;
    }
    return n;
  }

  static CUDADevice getDevice(int i)
  {
    return CUDADevice(i);
  }

  static void setCurrentDevice(const CUDADevice& device)
  {
    // TODO: set curr device
  }
};

// CUDA device memory ptr
template <class T>
class dev_ptr
{
  T* ptr_;
  int size_;

  void freePtr() { cudaFree(ptr_); }

public:
  dev_ptr() : ptr_(0), size_(0) {}
  explicit dev_ptr(int size) : size_(0) { set(NULL, size); }
  dev_ptr(T* ptr, int size) : size_(0) { set(ptr, size); }
  ~dev_ptr() { freePtr(); }

  bool set(T* ptr, int size)
  {
    if(size_ != 0)
      freePtr();
    size_ = size * sizeof(T);
    if(cudaSuccess != cudaMalloc(&ptr_, size_))
    {
      std::cerr << "Cannot alloc device memory of size" << size_ << std::endl;
      return false;
    }
    if(ptr != NULL && cudaSuccess != cudaMemcpy(ptr_, ptr, size_, cudaMemcpyHostToDevice))
    {
      std::cerr << "Cannot copy host mem to device of size" << size_ << std::endl;
      return false;
    }
    return true;
  }

  int sizeInBytes() { return size_; }
  int sizeInType() { return size_/sizeof(T); }
  void setSize(int size) { set(NULL, size); }

  void toHost(T* hptr) { cudaMemcpy(hptr, ptr_, size_, cudaMemcpyDeviceToHost); }
  T* ptr() { return ptr_; }
};

template<int size>
class RandState : public dev_ptr<curandState>
{
public:
  explicit RandState(int seed) : dev_ptr<curandState>(size) { setupRandStates(ptr(), size, seed); }
};


template<int numRows, int numCols, typename T>
class BaseGetter
{
public:
  __device__
  static T get(const T* ptr)
  {
    return ptr[0];
  }
};

template<int numRows, int numCols, typename T = float, typename Get = BaseGetter<numRows, numCols, T> >
class Matrix
{
  dev_ptr<T> dptr_; // device mem ptr
public:
  Matrix() { dptr_.setSize(numRows * numCols); }
  Matrix(T * ptr) { dptr_.set(ptr, numRows * numCols); }

  // fill with specific value
  void fill(T val) { fillMtxf(dptr_.ptr(), dptr_.sizeInType(), val); }
  void zeros() { fillMtxf(dptr_.ptr(), dptr_.sizeInType(), 0.0); }
  void ones() { fillMtxf(dptr_.ptr(), dptr_.sizeInType(), 1.0); }
  void randu(RandState<numRows * numCols>& state) { fillRand(dptr_.ptr(), numRows * numCols, state.ptr()); }

  // ops with type T
  Matrix<numRows, numCols, T>& operator  =(T val) { fill(val); return *this; }
  Matrix<numRows, numCols, T>& operator +=(T val) { addMtxf(dptr_.ptr(), dptr_.sizeInType(), val); return *this; }
  Matrix<numRows, numCols, T>& operator +=(Matrix<numRows, numCols, T>& val)
  { addMtxMtxf(dptr_.ptr(), val.dptr_.ptr(), dptr_.sizeInType()); return *this; }
  Matrix<numRows, numCols, T>& operator -=(T val) { addMtxf(dptr_.ptr(), dptr_.sizeInType(), -val); return *this; }
  Matrix<numRows, numCols, T>& operator *=(T val) { mulMtxf(dptr_.ptr(), dptr_.sizeInType(), val); return *this; }
  Matrix<numRows, numCols, T>& operator /=(T val) { mulMtxf(dptr_.ptr(), dptr_.sizeInType(), 1/val); return *this; }

  void print()
  {
    T * hptr = new T[numRows*numCols];
    dptr_.toHost(hptr);

    for(int i = 0; i < numRows; ++i)
    {
      for(int j = 0; j < numCols; ++j)
	std::cout << hptr[j*numRows + i] << "\t";
      std::cout << std::endl;
    }
    delete hptr;
  }

  T* devPtr() { return dptr_.ptr(); }
};

template<int numRows, typename T = float>
class RVector : Matrix<numRows, 1, T>
{
};

template<int numCols, typename T = float>
class CVector : Matrix<1, numCols, T>
{
};

#endif //_CUDA_UTILS_H_
