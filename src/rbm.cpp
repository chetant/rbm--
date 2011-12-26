#include <iostream>
#include "rbm.hpp"

using namespace std;

int main(int argc, char * argv[])
{

  RBM<3, 2> rbm;

  cout << "Hello!" << endl;
  rbm.printConfig();
}
