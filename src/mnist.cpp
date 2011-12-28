#include <stdint.h>
#include <iomanip>
#include "mnist.hpp"

using namespace std;
using boost::filesystem::path;

namespace mnist
{
  void Data::load()
  {
    path imgFname;
    path labFname;
    if(type == TrainingSet)
    {
      imgFname = rootPath / "train-images-idx3-ubyte.gz";
      labFname = rootPath / "train-labels-idx1-ubyte.gz";
    }
    else
    {
      imgFname = rootPath / "t10k-images-idx3-ubyte.gz";
      labFname = rootPath / "train-labels-idx1-ubyte.gz";
    }

    ifstream file(imgFname.c_str(), ios_base::in | ios_base::binary);
    imgStream.push(gzip_decompressor());
    imgStream.push(file);

    uint32_t tmp;
    imgStream >> tmp;
    cout << "0x" << hex << tmp << endl;
  }
}
