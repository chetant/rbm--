#ifndef _RBM_H_
#define _RBM_H_

#include <iostream>
#include <cuda_runtime_api.h>
#include <boost/filesystem.hpp>
#include "cuda_utils.hpp"

using boost::filesystem::path;

// #define Matrix arma::mat::fixed
// #define UMatrix arma::umat::fixed
// #define Vector arma::vec::fixed
// #define UVector arma::uvec::fixed
// #define RVector arma::rowvec::fixed
// #define Cube arma::cube::fixed
// #define Span arma::span

#define sigmoid(x) (1 / (1 + exp(-x)))

// #define dump(x) {std::cout << #x << std::endl;std::cout << x << std::endl;}
// #define dumpline(x) {std::cout << #x << x << std::endl;}
// #define dump2(x, y) {std::cout << x << std::endl;std::cout << y << std::endl;}

#define dump(x)
#define dumpline(x)
#define dump2(x, y)

template<int numVisible, int numSamples>
class BinPVisible
{
protected:
  Matrix<numSamples, numVisible> vs;

  enum { numNodes = numVisible };

public:
  void sampleVisible(const RVector<numVisible>& v)
  {
    // TODO: sample a bunch of vs from v(real valued)
  }
};

template<int numHidden, int numSamples, class Lower>
class BinHidden
{
protected:
  Matrix<Lower::numNodes, numHidden> weights;
  RVector<Lower::numNodes> visBias;
  RVector<numHidden> hidBias;

  Matrix<numSamples, numHidden> hs;
public:

  void bottomUp()
  {
  }

  void topDown()
  {
  }

};

template<int numVisible, int numHidden, 
	 int cdn = 10, int numSamples = 100, 
	 class VisClass = BinPVisible<numVisible, numSamples>, 
	 class HidClass = BinHidden<numHidden, numSamples, VisClass>
	 >
class RBM
{
protected:
  double learnRate;

  // template<int n, int i>
  // inline void sample(Matrix<n, i>& res, Matrix<n, i>& ps)
  // {
  //   res.randu();
  //   #pragma omp parallel for
  //   for(int j = 0; j < n; ++j)
  //     #pragma omp parallel for
  //     for(int k = 0; k < i; ++k)
  // 	res(j, k) = (res(j, k) < ps(j, k))? 1.0 : 0.0;
  // }

  // template<int n, int i>
  // inline void sampleSig(Matrix<n, i>& res, Matrix<n, i>& ps)
  // {
  //   res.randu();
  //   #pragma omp parallel for
  //   for(int j = 0; j < n; ++j)
  //     #pragma omp parallel for
  //     for(int k = 0; k < i; ++k)
  // 	res(j, k) = (res(j, k) < sigmoid(ps(j, k)))? 1.0 : 0.0;
  // }

  // template<int n>
  // inline void getVPsGivenHs(Matrix<n, numVisible>& ps,  Matrix<n, numHidden>& hs)
  // {
  //   ps = arma::repmat(visBias, n, 1) + hs * weightsT;
  // }

  // template<int n>
  // inline void getHPsGivenVs(Matrix<n, numHidden>& ps,  Matrix<n, numVisible>& vs)
  // {
  //   ps = arma::repmat(hidBias, n, 1) + vs * weights;
  // }

  // template<int n>
  // inline void getCorr(Matrix<numVisible, numHidden>& corrs, Matrix<n, numVisible>& vs, Matrix<n, numHidden>& hs)
  // {
  //   corrs = vs.t() * hs;
  //   // #pragma omp parallel for
  //   // for(int i = 0; i < numVisible; ++i)
  //   //   #pragma omp parallel for
  //   //   for(int j = 0; j < numHidden; ++j)
  //   //     #pragma omp parallel for
  //   // 	for(int k = 0; k < n; ++k)
  //   // 	  corrs(i, j) += vs(k, i) * hs(k, j);
  // }

public:
  RBM(double lRate = 0.07, double stdDev = 0.1)
  {
    // weights.randn();
    // weights *= stdDev;
    // weightsT = weights.t();
    // visBias.zeros();
    // hidBias.zeros();
    
    learnRate = lRate;
  }

  // RBM(const path& filename) { load(filename); }

  // template<int n>
  // void trainBatch(Matrix<n, numVisible>& batch, double& e)
  // {
  //   Matrix<n, numVisible> ovs;
  //   Matrix<n, numVisible> err;
  //   Matrix<n, numVisible> vs;
  //   Matrix<n, numVisible> pvs;

  //   Matrix<n, numHidden>  hs;
  //   Matrix<n, numHidden>  phs;

  //   Matrix<numVisible, numHidden> posCorr;
  //   Vector<numVisible> posVCorr;
  //   Vector<numHidden>  posHCorr;

  //   Matrix<numVisible, numHidden> negCorr;
  //   Vector<numVisible> negVCorr;
  //   Vector<numHidden>  negHCorr;

  //   Matrix<numVisible, numHidden> delWt;
  //   Matrix<numHidden, numVisible> delWtT;

  //   double epsilon = learnRate / n;

  //   dump(epsilon);

  //   for(int i = 0; i < numSamples; ++i)
  //   {
  //     sample<n, numVisible>(vs, batch);
  //     ovs = vs;
  //     dump(vs);
  //     getHPsGivenVs<n>(phs, vs);
  //     dump(phs);
  //     sampleSig<n, numHidden>(hs, phs);
  //     dump(hs);
  //     getCorr<n>(posCorr, vs, hs);
  //     dump(posCorr);
      
  //     for(int j = 0; j < cdn; ++j)
  //     {
  // 	getVPsGivenHs<n>(pvs, hs);
  // 	dump(pvs);
  // 	sampleSig<n, numVisible>(vs, pvs);
  // 	dump(vs);
  // 	getHPsGivenVs<n>(phs, vs);
  // 	dump(phs);
  // 	sampleSig<n, numHidden>(hs, phs);
  // 	dump(hs);
  //     }
  //     getCorr<n>(negCorr, vs, hs);
  //     dump(negCorr);

  //     delWt = epsilon * (posCorr - negCorr);
  //     dump(delWt);
  //     delWtT = delWt.t();
  //     weights += delWt;
  //     weightsT += delWtT;
  //     dump(weights);

  //     err = ovs - vs;
  //     dump(ovs);
  //     dump(vs);
  //     dump(err);
  //     e = arma::accu(err % err);
  //     dumpline(e);
  //   }
  // }

  // template<int n>
  // void reconstruct(RVector<numVisible>& newv, RVector<numVisible>& v)
  // {
  //   Matrix<n, numVisible> vs = repmat(v, n, 1);;
  //   Matrix<n, numVisible> pvs;
  //   Matrix<n, numHidden>  hs;
  //   Matrix<n, numHidden>  phs;
    
  //   getHPsGivenVs<n>(phs, vs);
  //   sampleSig<n, numHidden>(hs, phs);
  //   getVPsGivenHs<n>(pvs, hs);
  //   sampleSig<n, numVisible>(vs, pvs);
  //   newv = arma::sum(vs) / n;
  // }

  // bool save(const path& filename)
  // {
  //   std::ofstream outf(filename.c_str(), std::ios_base::out | std::ios_base::binary);
  //   weights.save(outf);
  //   visBias.save(outf);
  //   hidBias.save(outf);

  //   outf.write((char *)&cdn, sizeof(int));
  //   outf.write((char *)&numSamples, sizeof(int));
  //   outf.write((char *)&learnRate, sizeof(double));
  //   return true;
  // }

  // bool load(const path& filename)
  // {
  //   std::ifstream inf(filename.c_str(), std::ios_base::in | std::ios_base::binary);
  //   weights.load(inf);
  //   visBias.load(inf);
  //   hidBias.load(inf);

  //   inf.read((char *)&cdn, sizeof(int));
  //   inf.read((char *)&numSamples, sizeof(int));
  //   inf.read((char *)&learnRate, sizeof(double));

  //   weightsT = weights.t();
  // }

  double getLearnRate() { return learnRate; }
  void setLearnRate(double n) { learnRate = n; }

  // const Matrix<numVisible, numHidden> & getWeights()
  // { return weights; }

  void printConfig()
  {
    std::cout << "Visible:" << numVisible << ", Hidden:" << numHidden << std::endl;
  }
};

#endif
