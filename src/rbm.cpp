#include <iostream>
#include <boost/thread.hpp>
#include "rbm.hpp"
#include "mnist.hpp"
#include "gnuplot_i.hpp"

using namespace std;
using boost::thread;

const int cdn = 10; // number of gibbs sampling to sample (v.h)inf
const double learnRate = 0.07; // epsilon
const int numSamples = 150; // number of samples to draw from given visual input
const int numTrainExs = 5; // number of training examples in a batch
const int numBatches = 200; // number of training batches to run
const int numVisible = 28*28; // number of visual units
const int numHidden = 100; // number of hidden units

// int dream()
// {
//   const int n = 100;
//   RBM<numVisible, numHidden> rbm("test.rbm");

//   RVector<numVisible> rv;
//   RVector<numVisible> v;

//   Matrix<n, numVisible> buffer;

//   Gnuplot plot("image");

//   while(true)
//   {
//     // see low noise
//     v.randn();
//     v *= 0.1;

//     // Fill up buffer with what the rbm thinks it sees
//     rbm.reconstruct<1>(rv, v);
//     buffer.row(0) = rv;
//     for(int i = 1; i < n; i++)
//     {
//       v = buffer.row(i-1);
//       rbm.reconstruct<1>(rv, v);
//       buffer.row(i) = rv;
//     }

//     v = (255.0/n) * sum(buffer);

//     // plot dream
//     plot.plot_image(v.memptr(), 28, 28, "Dream sequence");
//     boost::this_thread::sleep(boost::posix_time::milliseconds(200));
//     plot.reset_plot();
//   }
//   return 0;
// }

int learn()
{
  // Setup RBM
  RBM<numVisible, numHidden> rbm;
  // rbm.setCdN(cdn);
  // rbm.setNumSamples(numSamples);
  // rbm.setLearnRate(learnRate);

  return 0;

  // Load the MNIST dataset
  mnist::Data trainSet(mnist::TrainingSet, "/saiko/data/digits");
  if(!trainSet.load())
  {
    cout << "Cannot load training set!" << endl;
    return 1;
  }

  // // Get the batch images
  // Matrix<numTrainExs, numVisible> test;
  // RVector<numVisible> v;
  // int label;
  // double err;
  // for(int i = 0; i < numBatches; ++i)
  // {
  //   for(int j = 0; j < numTrainExs;)
  //   {
  //     if(!trainSet.loadNext<numVisible>(v, label))
  //     {
  // 	cout << "Cannot load image!" << endl;
  // 	return 1;
  //     }
  //     if(label != 0)
  // 	continue;

  //     test.row(j) = v;
  //     ++j;
  //   }
  //   rbm.trainBatch<numTrainExs>(test, err);
  //   cout << "Batch:" << i << ", Error:" << err << endl;
  // }

  // rbm.printConfig();
  // rbm.save("test.rbm");

  // dump2("Final", rbm.getWeights());
  // cout << "Final" << endl;
  // cout << rbm.getWeights() << endl;

  return 0;
}

int main(int argc, char * argv[])
{
  int numDev = CUDASystem::getNumDevices();
  cout << "Number of CUDA devices found:" << numDev << endl;
  for(int i = 0; i < numDev; ++i)
  {
    CUDADevice dev = CUDASystem::getDevice(i);
    printf("Device %d: %s, MP: %d, Compute:%1.1f, WarpSize: %d\n", i, dev.name().c_str(), dev.multiProcs(), dev.computeCaps(), dev.warpSize());
  }

  // return learn();
  // return dream();

  Matrix<3, 2> testm;
  // testm.zeros();
  testm = 51;
  testm *= 2;
  testm.print();

}
