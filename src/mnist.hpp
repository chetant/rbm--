#ifndef _MNIST_H_
#define _MNIST_H_

#include <fstream>
#include <iostream>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
using namespace boost::iostreams;

namespace mnist
{
  enum SetType { TrainingSet, TestSet };

  class Data
  {
    SetType type;
    path rootPath;
    filtering_streambuf<input> imgStream;
    filtering_streambuf<input> labStream;
  public:
    Data(SetType type, path pth) : rootPath(pth), type(type) {}
    void load();
  };
}
#endif //_MNIST_H_
