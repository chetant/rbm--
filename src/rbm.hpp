#ifndef _RBM_H_
#define _RBM_H_

#include <iostream>
#include <cmath>
#include <armadillo>

#define Matrix arma::mat::fixed
#define Vector arma::vec::fixed
#define RVector arma::rowvec::fixed
#define Cube arma::cube::fixed
#define Span arma::span

#define sigmoid(x) (1 / (1 + exp(-x)))

template<int numVisible, int numHidden>
class RBM
{
protected:
  Matrix<numVisible, numHidden> weights;
  Matrix<numHidden, numVisible> weightsT;
  RVector<numVisible> visBias;
  RVector<numHidden> hidBias;

  int cdn;
  int numSamples;

  template<int n, int i>
  inline void sample(Matrix<n, i>& res, Matrix<n, i>& ps)
  {
    res.randu();
    #pragma omp parallel for
    for(int j = 0; j < n; ++j)
      #pragma omp parallel for
      for(int k = 0; k < i; ++k)
	res(j, k) = (res(j, k) < ps(j, k))? 1.0 : 0.0;
  }

  template<int n, int i>
  inline void sampleSig(Matrix<n, i>& res, Matrix<n, i>& ps)
  {
    res.randu();
    #pragma omp parallel for
    for(int j = 0; j < n; ++j)
      #pragma omp parallel for
      for(int k = 0; k < i; ++k)
	res(j, k) = (res(j, k) < sigmoid(ps(j, k)))? 1.0 : 0.0;
  }

  template<int n>
  inline void getVPsGivenHs(Matrix<n, numVisible>& ps,  Matrix<n, numHidden>& hs)
  {
    ps = arma::repmat(visBias, n, 1) + hs * weightsT;
  }

  template<int n>
  inline void getHPsGivenVs(Matrix<n, numHidden>& ps,  Matrix<n, numVisible>& vs)
  {
    ps = arma::repmat(hidBias, n, 1) + vs * weights;
  }

  template<int n>
  inline void getCorr(Matrix<numVisible, numHidden>& corrs, Matrix<n, numVisible>& vs, Matrix<n, numHidden>& hs)
  {
    Cube<numVisible, numHidden, n> vm;
    Cube<numVisible, numHidden, n> hm;
    Matrix<numVisible, n> vsT = vs.t();

    // vm(s, Span::all)
  }

public:
  RBM(int numSamples = 10, int cdLearnLoops = 10, double stdDev = 0.1)
  {
    weights.randn();
    weights *= stdDev;
    weightsT = weights.t();
    visBias.zeros();
    hidBias.zeros();
    
    cdn = cdLearnLoops;
    this->numSamples = numSamples;
  }

  template<int n>
  void trainBatch(Matrix<n, numVisible>& batch)
  {
    Matrix<n, numVisible> vs;
    Matrix<n, numVisible> pvs;

    Matrix<n, numHidden>  hs;
    Matrix<n, numHidden>  phs;

    Matrix<numVisible, numHidden> posCorr;
    Vector<numVisible> posVCorr;
    Vector<numHidden>  posHCorr;

    Matrix<numVisible, numHidden> negCorr;
    Vector<numVisible> negVCorr;
    Vector<numHidden>  negHCorr;

    for(int i = 0; i < numSamples; ++i)
    {
      sample<n, numVisible>(vs, batch);
      getHPsGivenVs<n>(phs, vs);
      sampleSig<n, numHidden>(hs, phs);
      getCorr<n>(posCorr, vs, hs);
      
      for(int j = i; j < cdn; ++j)
      {
	getVPsGivenHs<n>(pvs, hs);
	sampleSig<n, numVisible>(vs, pvs);
	getHPsGivenVs<n>(phs, vs);
	sampleSig<n, numHidden>(hs, phs);
      }
      getCorr<n>(negCorr, vs, hs);
    }
  }

  int getCdN() { return cdn; }
  void setCdN(int n) { cdn = n; }

  int getNumSamples() { return numSamples; }
  void setNumSamples(int n) { numSamples = n; }

  void printConfig()
  {
    std::cout << "Visible:" << numVisible << ", Hidden:" << numHidden << std::endl;
  }
};

#endif
