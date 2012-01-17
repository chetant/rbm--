#ifndef _CUDA_UTILS_H_
#define _CUDA_UTILS_H_

#include <iostream>
#include <string>

#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

using std::cerr;
using std::endl;
using std::string;

void sampleVis(float * v, float * vs, curandState * randStates, int numSamples, int numVisible);
void sampleHid(float * hs, curandState * randStates, int numSamples, int numHidden);

void setupRandStates(curandState * state, int size, int seed);
void fillRand(float * data, int size, curandState * state);

void fillMtxf(float * ptr, int size, float val);
void mulMtxf(float * ptr, int size, float val);
void addMtxf(float * ptr, int size, float val);

void addMtxMtxf(float * toptr, const float * fromptr, int size);
void addMtxRVecf(float * toptr, const float * fromptr, int numRows, int numCols);
void subMtxMtxf(float * toptr, const float * fromptr, int size);
void subMtxRVecf(float * toptr, const float * fromptr, int numRows, int numCols);
void mulMtxMtxf(float * toptr, const float * fromptr, int size);
void gtMtxMtxf(float * toptr, const float * fromptr, int size);

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
  static CUDADevice currDevice;

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
    currDevice = device;
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
  void fromHost(T* hptr) { cudaMemcpy(ptr_, hptr, size_, cudaMemcpyHostToDevice); }
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
  Matrix<numRows, numCols, T>& operator -=(T val) { addMtxf(dptr_.ptr(), dptr_.sizeInType(), -val); return *this; }
  Matrix<numRows, numCols, T>& operator *=(T val) { mulMtxf(dptr_.ptr(), dptr_.sizeInType(), val); return *this; }
  Matrix<numRows, numCols, T>& operator /=(T val) { mulMtxf(dptr_.ptr(), dptr_.sizeInType(), 1/val); return *this; }

  Matrix<numRows, numCols, T>& operator +=(Matrix<numRows, numCols, T>& val)
  { addMtxMtxf(dptr_.ptr(), val.dptr_.ptr(), dptr_.sizeInType()); return *this; }
  Matrix<numRows, numCols, T>& operator +=(Matrix<numCols, 1, T>& val)
  { addMtxRVecf(dptr_.ptr(), val.devPtr(), numRows, numCols); return *this; }
  Matrix<numRows, numCols, T>& operator -=(Matrix<numRows, numCols, T>& val)
  { subMtxMtxf(dptr_.ptr(), val.dptr_.ptr(), dptr_.sizeInType()); return *this; }
  Matrix<numRows, numCols, T>& operator -=(Matrix<numCols, 1, T>& val)
  { subMtxRVecf(dptr_.ptr(), val.devPtr(), numRows, numCols); return *this; }
  Matrix<numRows, numCols, T>& operator %=(Matrix<numRows, numCols, T>& val)
  { gtMtxMtxf(dptr_.ptr(), val.dptr_.ptr(), dptr_.sizeInType()); return *this; }

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

  bool save(std::ostream outf)
  {
    T * hptr = new T[numRows*numCols];
    dptr_.toHost(hptr);

    int nrows = numRows;
    int ncols = numCols;
    outf.write((const char *)&nrows, sizeof(nrows));
    outf.write((const char *)&ncols, sizeof(ncols));
    outf.write((const char *)hptr, numRows * numCols * sizeof(T));

    delete hptr;
    return outf.fail();
  }

  bool load(std::istream inf)
  {
    T * hptr = new T[numRows*numCols];

    int nrows, ncols;
    inf.read((char *)&nrows, sizeof(nrows));
    inf.read((char *)&ncols, sizeof(ncols));
    if(nrows != numRows || ncols != numCols)
      return false;
    inf.read((char *)hptr, numRows * numCols * sizeof(T));

    dptr_.fromHost(hptr);

    delete hptr;
    return inf.fail();
  }
};

template<int numRows, typename T = float>
class RVector : public Matrix<numRows, 1, T>
{
};

template<int numCols, typename T = float>
class CVector : public Matrix<1, numCols, T>
{
};

class CuBLAS
{
private:
  CuBLAS(const CuBLAS&);
  CuBLAS& operator=(const CuBLAS&);
protected:
  cublasHandle_t handle;
public:
  CuBLAS()
  {
    if(CUBLAS_STATUS_SUCCESS != cublasCreate(&handle))
    {
      cerr << "ERROR: Cannot init CuBLAS library!" << endl;
    }
  }
  ~CuBLAS() { cublasDestroy(handle); }

  template<int m, int n, int k>
  void mul(Matrix<m, k>& a, Matrix<k, n>& b, Matrix<m, n>& c)
  {
    float ac = 1.0;
    float bc = 0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &ac, a.devPtr(), m, b.devPtr(), k, &bc, c.devPtr(), m);
  }

  template<int m, int n, int k>
  void mulT(Matrix<m, k>& a, Matrix<n, k>& b, Matrix<m, n>& c)
  {
    float ac = 1.0;
    float bc = 0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &ac, a.devPtr(), m, b.devPtr(), n, &bc, c.devPtr(), m);
  }

  template<int m, int n, int k>
  void mulT(Matrix<k, m>& a, Matrix<k, n>& b, Matrix<m, n>& c)
  {
    float ac = 1.0;
    float bc = 0;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &ac, a.devPtr(), k, b.devPtr(), k, &bc, c.devPtr(), m);
  }
};

#endif //_CUDA_UTILS_H_
