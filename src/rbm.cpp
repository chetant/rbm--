#include <iostream>
#include "rbm.hpp"

using namespace std;

int main(int argc, char * argv[])
{
  const int numSamples = 2;
  RBM<3, 2> rbm;
  rbm.setCdN(10);
  rbm.setNumSamples(25);
  rbm.setLearnRate(0.07);

  RVector<3> v;
  v << 1 << 0 << 1;
  
  // Matrix<numSamples, 3> test;
  // test.randu();
  Matrix<numSamples, 3> test = repmat(v, numSamples, 1);

  dump2("Init", rbm.getWeights());
  dump(rbm.getWeights());

  rbm.trainBatch<numSamples>(test);

  rbm.printConfig();
  dump2("Final", rbm.getWeights());

  RVector<3> v2;
  v2 << 1 << 1 << 0;
  RVector<3> v3;
  rbm.reconstruct<10>(v3, v2);
  dump(v3);
  cout << v3 << endl;
}
