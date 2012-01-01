#include <iostream>
#include "rbm.hpp"
#include "mnist.hpp"

using namespace std;

template<int n, int numVisible, int numHidden>
void dream(RBM<numVisible, numHidden>& rbm)
{
  // TODO: randomize vs, and perform gibbs sampling, average the last n inputs and plot them
  RVector<numVisible> v;
  v.randn();
  v *= 0.1;

  Matrix<numVisible, n> buffer;

  // Fill up buffer
  buffer.row(0) = rbm.reconstruct<1>(v);
  for(int i = 1; i < n; i++)
    buffer.row(i) = rbm.reconstruct<1>(buffer.row(i-1));

  int currI = 0;
  v = sum(buffer);
}

int main(int argc, char * argv[])
{
  const int cdn = 10; // number of gibbs sampling to sample (v.h)inf
  const double learnRate = 0.07; // epsilon
  const int numSamples = 100; // number of samples to draw from given visual input
  const int numTrainExs = 100; // number of training examples in a batch
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

  // Get the batch images
  Matrix<numTrainExs, numVisible> test;
  RVector<numVisible> v;
  int label;
  for(int i = 0; i < numTrainExs;)
  {
    if(!trainSet.loadNext<numVisible>(v, label))
    {
      cout << "Cannot load image!" << endl;
      return 1;
    }
    if(label != 0)
      continue;

    test.row(i) = v;
    ++i;
  }

  rbm.trainBatch<numTrainExs>(test);
  rbm.printConfig();
  rbm.save("test.rbm");

  // dump2("Final", rbm.getWeights());
  // cout << "Final" << endl;
  // cout << rbm.getWeights() << endl;

  // RVector<3> v2;
  // v2 << 1 << 1 << 0;
  // RVector<3> v3;
  // rbm.reconstruct<10>(v3, v2);
  // dump(v3);
  // cout << v3 << endl;

  return 0;
}
