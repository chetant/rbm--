#ifndef _RBM_H_
#define _RBM_H_

#include <iostream>
#include <armadillo>

template<int numVisible, int numHidden>
class RBM
{
protected:
  arma::mat::fixed<numVisible, numHidden> weights;
  arma::mat::fixed<numHidden, numVisible> weightsT;

public:
  RBM(double stdDev = 0.1)
  {
    weights.randn();
    weightsT = weights.t();
  }

  void printConfig()
  {
    std::cout << "Visible:" << numVisible << ", Hidden:" << numHidden << std::endl;
  }
};

#endif

