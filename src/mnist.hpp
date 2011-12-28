#ifndef _MNIST_H_
#define _MNIST_H_

#include <fstream>
#include <iostream>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/filesystem.hpp>

namespace mnist
{
  enum SetType { TrainingSet, TestSet };

  class Data
  {
    SetType type;
    boost::filesystem::path rootPath;
    boost::iostreams::filtering_streambuf<boost::iostreams::input> imgStream;
    boost::iostreams::filtering_streambuf<boost::iostreams::input> labStream;
  public:
    Data(SetType type, boost::filesystem::path pth) : rootPath(pth), type(type) {}
    bool load();
  };
}
#endif //_MNIST_H_
