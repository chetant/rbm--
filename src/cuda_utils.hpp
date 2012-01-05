#ifndef _CUDA_UTILS_H_
#define _CUDA_UTILS_H_

// CUDA device memory ptr
template <class T>
class dev_ptr
{
  T* ptr_;
  int size_;

  T* ptr() { return ptr; }

  void freePtr() { cudaFree(ptr_); }

public:
  dev_ptr() : ptr_(0), size_(0) {}
  dev_ptr(T* ptr, int size) : size_(0) { set(ptr, size); }
  ~dev_ptr() { freePtr(); }

  bool set(T* ptr, int size)
  {
    if(size_ != 0)
      freePtr();
    size_ = size;
    cudaMalloc(&ptr_, size_); 
    cudaMemcpy(ptr_, ptr, size, cudaMemcpyHostToDevice);
  }

  int size() { return size_; }

  void toHost(T* hptr) { cudaMemcpy(hptr, ptr_, size_, cudaMemcpyDeviceToHost); }
};

template<int numRows, int numCols>
class Matrix
{
  dev_ptr<float> dptr_; // device mem ptr
public:
  randu(); //  fill with 0-1 uniform distribution
};

template<int numRows>
class RVector : Matrix<numRows, 1>
{

};

#endif //_CUDA_UTILS_H_
