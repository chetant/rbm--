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

// void dream(RBM<numVisible, numHidden>& rbm)
// {
//   // TODO: randomize vs, and perform gibbs sampling, average the last n inputs and plot them
//   RVector<numVisible> v;
//   v.randn();
//   v *= 0.1;

//   Matrix<n, numVisible> buffer;

//   // Fill up buffer
//   buffer.row(0) = rbm.reconstruct<1>(v);
//   for(int i = 1; i < n; i++)
//     buffer.row(i) = rbm.reconstruct<1>(buffer.row(i-1));

//   int currI = 0;
//   v = sum(buffer);
// }

static boost::mutex mut;

// void chart(RVector<numVisible>& v)
// {
//   Gnuplot plot("image");

//   {  
//     boost::lock_guard<boost::mutex> lockV(mut);
//     plot.plot_image(v.memptr(), 28, 28, "Dream sequence");
//     plot.reset_plot();
//     sleep(1);
//   }
  
//   sleep(10);
// }

int dream()
{
  const int n = 100;
  RBM<numVisible, numHidden> rbm("test.rbm");

  RVector<numVisible> rv;
  RVector<numVisible> v;

  Matrix<n, numVisible> buffer;

  Gnuplot plot("image");

  while(true)
  {
    // see low noise
    v.randn();
    v *= 0.1;

    // Fill up buffer with what the rbm thinks it sees
    rbm.reconstruct<1>(rv, v);
    buffer.row(0) = rv;
    for(int i = 1; i < n; i++)
    {
      v = buffer.row(i-1);
      rbm.reconstruct<1>(rv, v);
      buffer.row(i) = rv;
    }

    v = (255.0/n) * sum(buffer);

    // plot dream
    plot.plot_image(v.memptr(), 28, 28, "Dream sequence");
    boost::this_thread::sleep(boost::posix_time::milliseconds(200));
    plot.reset_plot();
  }
  return 0;
}

// int dream()
// {
//   const int n = 10;
//   RBM<numVisible, numHidden> rbm("test.rbm");

//   RVector<numVisible> rv;
//   RVector<numVisible> v;
//   v.randn();
//   v *= 0.1;

//   Matrix<n, numVisible> buffer;

//   // Fill up buffer
//   rbm.reconstruct<1>(rv, v);
//   buffer.row(0) = rv;
//   for(int i = 1; i < n; i++)
//   {
//     v = buffer.row(i-1);
//     rbm.reconstruct<1>(rv, v);
//     buffer.row(i) = rv;
//   }

//   v = (255.0/n) * sum(buffer);

//   // boost::unique_lock<boost::mutex> lockV(mut);
//   // thread drawThread(chart, boost::ref(v));
//   // sleep(5); lockV.unlock();

//   // drawThread.join();
  
//   Gnuplot plot("image");
//   int currI = 0;
//   int lastI = n-1;
//   while(true)
//   {
//     v = buffer.row(lastI);
//     rbm.reconstruct<1>(rv, v);
//     buffer.row(currI) = rv;

//     ++currI;  currI %= n;
//     ++lastI;  lastI %= n;

//     v = sum(buffer);
//     plot.plot_image(v.memptr(), 28, 28, "Dream sequence");
//     boost::this_thread::sleep(boost::posix_time::milliseconds(125));
//     plot.reset_plot();
//   }
//   return 0;
// }

int learn()
{
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
  double err;
  for(int i = 0; i < numBatches; ++i)
  {
    for(int j = 0; j < numTrainExs;)
    {
      if(!trainSet.loadNext<numVisible>(v, label))
      {
	cout << "Cannot load image!" << endl;
	return 1;
      }
      if(label != 0)
	continue;

      test.row(j) = v;
      ++j;
    }
    rbm.trainBatch<numTrainExs>(test, err);
    cout << "Batch:" << i << ", Error:" << err << endl;
  }

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

int main(int argc, char * argv[])
{
  // return learn();
  return dream();
}
