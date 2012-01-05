#ifndef _CUDA_UTILS_H_
#define _CUDA_UTILS_H_

#include <iostream>

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

void fillMtxf(float * ptr, int size, float val);
void mulMtxf(float * ptr, int size, float val);

void test();

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

template<int numRows, int numCols, typename T = float>
class Matrix
{
  dev_ptr<T> dptr_; // device mem ptr
public:
  Matrix() { dptr_.setSize(numRows * numCols); }
  Matrix(T * ptr) { dptr_.set(ptr, numRows * numCols); }

  // fill with specific value
  void fill(T val) { fillMtxf(dptr_.ptr(), dptr_.sizeInBytes(), val); }
  void zeros() { fillMtxf(dptr_.ptr(), dptr_.sizeInBytes(), 0.0); }
  void ones() { fillMtxf(dptr_.ptr(), dptr_.sizeInBytes(), 1.0); }

  // assignment with a type T is a fill
  Matrix<numRows, numCols>& operator =(T val)
  {
    fill(val);
    return *this;
  }

  Matrix<numRows, numCols>& operator *=(T val)
  {
    mulMtxf(dptr_.ptr(), dptr_.sizeInBytes(), val);
    return *this;
  }

  //  fill with 0-1 uniform distribution
  void randu()
  {
  }

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
