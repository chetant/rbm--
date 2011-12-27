#include <iostream>
#include "rbm.hpp"

using namespace std;

int main(int argc, char * argv[])
{

  RBM<3, 2> rbm(10, 100);

  Matrix<10, 3> test;
  test.randu();

  rbm.trainBatch<10>(test);

  cout << "Hello!" << endl;
  rbm.printConfig();
}
