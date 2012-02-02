#ifndef _RBM_H_
#define _RBM_H_

#include <ctime>
#include <iostream>
#include <cuda_runtime_api.h>
#include <boost/filesystem.hpp>
#include "cuda_utils.hpp"

using boost::filesystem::path;

#define sigmoid(x) (1 / (1 + exp(-x)))
#define MAXD(x, y) (((int)x) > ((int)y) ? ((int)x) : ((int)y))
#define MAX3(x, y, z) (MAXD(z, MAXD(x, y)))

#define dump(x) {std::cout << #x << std::endl;std::cout << x << std::endl;}
#define dumpline(x) {std::cout << #x << x << std::endl;}
#define dump2(x, y) {std::cout << x << std::endl;std::cout << y << std::endl;}

// #define dump(x)
// #define dumpline(x)
// #define dump2(x, y)

// template for a number type, to overload functions
template<int n> struct L { enum { value = n }; };
extern L<0> L0; extern L<1> L1; extern L<2> L2; extern L<3> L3; extern L<4> L4; extern L<5> L5; 

template<int numVisible, int numSamples>
class BinPVisible
{
protected:
  enum { numNodes = numVisible };
  enum { numInputs = numVisible };

  CuBLAS blas;

  Matrix<numSamples, numVisible> vs;
  dev_ptr<curandState> randStates;

  inline CuBLAS& getBlas()  { return blas; }
  inline Matrix<numSamples, numVisible>& getHsFromVs(CuBLAS&) { return vs;}
  inline Matrix<numSamples, numVisible>& getHs() { return vs; }

public:
  BinPVisible() : randStates(numSamples * numNodes) { setupRandStates(randStates.ptr(), numSamples * numNodes, time(NULL)); }

  void setSample(RVector<numVisible>& v)
  {
    // sample from v
    sampleVis(v.devPtr(), vs.devPtr(), randStates.ptr(), numSamples, numVisible);
    std::cout << "Vs:" << std::endl;
    vs.print();
  }

  double cdLearn(int, RVector<numInputs>&, int, double)
  {
    std::cout << "ERROR: too deep in the dbn!" << std::endl;
  }

  bool save(std::ostream& outf) {return true;}
  bool load(std::istream& inf)  {return true;}

  void printVis() { vs.print(); }
  Matrix<numSamples, numVisible>& getVis() { return vs; }
};

template<int N, int numHidden, int numSamples, class Lower, bool isOuterMost = false>
class BinHidden : public Lower
{
protected:
  enum { numNodes = numHidden };
  enum { numInputs = Lower::numInputs };

  Matrix<Lower::numNodes, numHidden> weights;
  RVector<Lower::numNodes> lowBias;
  RVector<numHidden> hidBias;

  Matrix<numSamples, numHidden> hs;
  dev_ptr<curandState> randStates;

  Matrix<Lower::numNodes, numHidden> pcorr;
  Matrix<Lower::numNodes, numHidden> ncorr;
  RVector<Lower::numNodes> lowBCorr;
  RVector<numHidden> hidBCorr;

  inline CuBLAS& getBlas()
  { return Lower::getBlas(); }

  template<int i, int j>
  inline void sample(Matrix<i, j>& m)
  {
    sampleHid(m.devPtr(), randStates.ptr(), i, j);
  }

  inline Matrix<numSamples, numHidden>& 
  getHsFromLs(Matrix<numSamples, Lower::numNodes>& ls, CuBLAS& blas)
  {
    blas.mul(ls, weights, hs);
    hs += hidBias;
    sample(hs);
    return hs;
  }

  inline Matrix<numSamples, Lower::numNodes>& 
  getLsFromHs(Matrix<numSamples, Lower::numNodes>& ls, CuBLAS& blas)
  {
    blas.mulT(hs, weights, ls);
    ls += lowBias;
    sample(ls);
    return ls;
  }

  inline Matrix<numSamples, numHidden>& getHsFromVs(CuBLAS& blas)
  {
    // before we do anything, we need hs from the lower level
    Matrix<numSamples, Lower::numNodes>& ls = Lower::getHsFromVs(blas);
    // get this layer's hs
    getHsFromLs(ls, blas);
    return hs;
  }

  inline Matrix<numSamples, numHidden>& getHs() { return hs; }

  inline void updateWeights(Matrix<Lower::numNodes, numHidden>& pcorr, 
			    Matrix<Lower::numNodes, numHidden>& ncorr, double epsilon)
  {
    // std::cout << "Before ws:" << std::endl;
    // weights.print();

    // std::cout << "p1:" << std::endl;
    // pcorr.print();
    // std::cout << "n1:" << std::endl;
    // ncorr.print();

    pcorr -= ncorr;
    // std::cout << "p2:" << std::endl;
    // pcorr.print();

    pcorr *= epsilon;
    // std::cout << "p3:" << std::endl;
    // pcorr.print();

    weights += pcorr;

    // std::cout << "After ws:" << std::endl;
    // weights.print();
  }

public:
  enum { currLevel = N };
  typedef Lower Next;

  BinHidden() : randStates(MAX3(numSamples * numNodes, numSamples * Lower::numNodes, numNodes * Lower::numNodes))
  {
    const int randSize = MAX3(numSamples * numNodes, numSamples * Lower::numNodes, numNodes * Lower::numNodes);
    int seed = time(NULL);
    setupRandStates(randStates.ptr(), randSize, seed);
    fillRandN(weights.devPtr(), Lower::numNodes * numHidden, randStates.ptr(), 0, 0.1);
    hidBias.zeros();
    lowBias.zeros();
  }

  double cdLearn(int level, RVector<numInputs>& v, int cdn, double epsilon)
  {
    if(level == N)
    {
      CuBLAS& blas = getBlas();
      std::cout << "I am number:" << N << std::endl;
      // sample from the v
      setSample(v);
      // now sample from vis layer to (level-1) layer
      getHsFromVs(blas);

      // std::cout << "Lower::getHs():" << std::endl;
      // Lower::getHs().print();

      // get pcorr
      Matrix<numSamples, Lower::numNodes>& ls = Lower::getHs();
      Matrix<numSamples, Lower::numNodes> pls = ls; // save a copy
      blas.mulT(ls, hs, pcorr);
      // now gibbs sample n times
      for(int i = 0; i < cdn; ++i)
      {
      	// from given hs, get ls
      	getHsFromLs(ls, blas);
	// std::cout << "cd" << i << " hs:" << std::endl;
	// hs.print();
      	// and go back up
      	getLsFromHs(ls, blas);
	// std::cout << "cd" << i << " ls:" << std::endl;
	// ls.print();
      }
      // finall get ncorr
      blas.mulT(ls, hs, ncorr);

      // update weights and biases
      updateWeights(pcorr, ncorr, epsilon);

      // std::cout << "pls:" << std::endl;
      // pls.print();

      std::cout << "ls:" << std::endl;
      ls.print();

      return 0.0;
    }
    else
      return Lower::cdLearn(level, v, cdn, epsilon);
  }

  bool saveDBN(path& outPath)
  {
    std::ofstream outf(outPath.c_str(), std::ios_base::out | std::ios_base::binary);
    save(outf);
    Lower::save(outf);
  }

  bool save(std::ostream& outf)
  {
    int nrows = Lower::numNodes;
    int ncols = numHidden;
    outf.write((const char *)&nrows, sizeof(nrows));
    outf.write((const char *)&ncols, sizeof(ncols));
    weights.save(outf);
    lowBias.save(outf);
    hidBias.save(outf);

    return outf.fail();
  }

  bool load(std::istream& inf)
  {
    int nrows, ncols;
    inf.read((char *)&nrows, sizeof(nrows));
    inf.read((char *)&ncols, sizeof(ncols));
    if(nrows != Lower::numNodes || ncols != numHidden)
      return false;
    weights.load(inf);
    lowBias.load(inf);
    hidBias.load(inf);

    return inf.fail();
  }
};

// finally a way to statically construct a DBN
template<int nNodes> struct VPBin 
{ template<int numSamples>  struct Type { typedef BinPVisible<nNodes, numSamples>  In; }; };
template<int nNodes> struct HBin   
{ template<bool isFirst, int N, int numSamples, typename L> struct Type { typedef BinHidden<N, nNodes, numSamples, L, isFirst> In; }; };

namespace
{
  // template to compute typelist length
  template<typename... Ts> struct TypeLen {};
  template<typename T, typename... Ts> struct TypeLen<T, Ts...> { enum { value = 1 + TypeLen<Ts...>::value }; };
  template<typename T> struct TypeLen<T> { enum { value = 1 }; };
  template<> struct TypeLen<> { enum { value = 0 }; };

  // helper templates to create a DBN
  template<bool isFirst, int N, int numSamples, typename... Ls> struct DBN_ {};
  template<int N, int numSamples, typename H, typename... Hs>
  struct DBN_<true, N, numSamples, H, Hs...> 
  { typedef typename H::template Type<true, N, numSamples, typename DBN_<false, N-1, numSamples, Hs...>::Object>::In Object; };
  template<int N, int numSamples, typename H, typename... Hs>
  struct DBN_<false, N, numSamples, H, Hs...> 
  { typedef typename H::template Type<false, N, numSamples, typename DBN_<false, N-1, numSamples, Hs...>::Object>::In Object; };
  template<int numSamples, typename V> 
  struct DBN_<false, 0, numSamples, V> { typedef typename V::template Type<numSamples>::In Object; };

  // helper template to give type of n'th entry in a DBN
  template<typename D, int i> struct TypeAt { typedef typename TypeAt<typename D::Next, i-1>::Type Type; };
  template<typename D> struct TypeAt<D, 0> { typedef D Type; };
}

template<int numSamples, typename... Ls> struct DBN {};
template<int numSamples, typename H, typename... Hs> 
struct DBN<numSamples, H, Hs...> { typedef typename DBN_<true, TypeLen<Hs...>::value, numSamples, H, Hs...>::Object Object; };

#endif
