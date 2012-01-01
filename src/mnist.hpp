#ifndef _MNIST_H_
#define _MNIST_H_

#include <stdint.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/filesystem.hpp>

#include "rbm.hpp"

using namespace std;
using boost::filesystem::path;
namespace io = boost::iostreams;

uint32_t read32u(io::filtering_istream& inf);
uint8_t read8u(io::filtering_istream& inf);

namespace mnist
{
  enum SetType { TrainingSet, TestSet };

  class Data
  {
    SetType type;
    path rootPath;
    io::filtering_istream imgStream;
    io::filtering_istream labStream;

    int numEntries;
    int numRows, numCols;
    int dataSize;

    // Currently loaded
    unique_ptr<uint8_t> imgData;
    uint8_t label;

    bool loadImgNLabFile(const path& imgFname, const path& labFname);

  public:
    Data(SetType type, const path& pth) : rootPath(pth), type(type) {}
    bool load();
    void displayImage();

    template<int numVisible>
    bool loadNext(RVector<numVisible>& vs, int& label)
    {
      if(!imgStream.good())
      	return false;
      if(numVisible != dataSize)
      	return false;

      imgStream.read((char *)imgData.get(), dataSize);
      uint8_t * ptr = imgData.get();
      for(int i = 0; i < numVisible; ++i)
      	  vs(i) = *ptr++ / 256.0;

      label = read8u(labStream);
      return true;
    }
  };
}
#endif //_MNIST_H_
