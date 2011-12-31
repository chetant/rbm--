#include <iostream>
#include "rbm.hpp"
#include "mnist.hpp"

using namespace std;

template<int n, int numVisible, int numHidden>
void dream(RBM<numVisible, numHidden>& rbm)
{
  // TODO: randomize vs, and perform gibbs sampling, average the last n inputs and plot them
}

int main(int argc, char * argv[])
{
  const int cdn = 10; // number of gibbs sampling to sample (v.h)inf
  const double learnRate = 0.07; // epsilon
  const int numSamples = 100; // number of samples to draw from given visual input
  const int numTrainExs = 4; // number of training examples in a batch
  const int numVisible = 28*28; // number of visual units
  const int numHidden = 50; // number of hidden units

  // Setup RBM
  RBM<numVisible, numHidden> rbm;
  rbm.setCdN(cdn);
  rbm.setNumSamples(numSamples);
  rbm.setLearnRate(learnRate);

  // Load the MNIST dataset
  mnist::Data trainSet(mnist::TrainingSet, "/saiko/data/digits");
  if(!trainSet.load())
  {
    cout << "Cannot load training set!" << endl;
    return 1;
  }
  // Get the first image
  RVector<numVisible> v;
  int label;
  if(!trainSet.loadNext<numVisible>(v, label))
  {
    cout << "Cannot load image!" << endl;
    return 1;
  }
  cout << "Image Label:" << label << endl;
  Matrix<numSamples, numVisible> test = repmat(v, numSamples, 1);

  dump2("Init", rbm.getWeights());
  dump(rbm.getWeights());

  rbm.trainBatch<numSamples>(test);

  rbm.printConfig();
  dump2("Final", rbm.getWeights());

  // RVector<3> v2;
  // v2 << 1 << 1 << 0;
  // RVector<3> v3;
  // rbm.reconstruct<10>(v3, v2);
  // dump(v3);
  // cout << v3 << endl;


  return 0;
}
