#include <iomanip>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file.hpp>
#include "mnist.hpp"

#define htonl(x) (((x)<<24) | (((x)<<8) & 0xFF0000) | (((x)>>8) & 0xFF00) | ((x)>>24))

uint32_t read32u(io::filtering_istream& inf)
{
    uint32_t tmp;
    inf.read((char *)&tmp, sizeof(tmp));
    return htonl(tmp);
}

uint8_t read8u(io::filtering_istream& inf)
{
    uint8_t tmp;
    inf.read((char *)&tmp, sizeof(tmp));
    return tmp;
}

namespace mnist
{
  bool Data::loadImgNLabFile(const path& imgFname, const path& labFname)
  {
    imgStream.reset();
    imgStream.push(io::gzip_decompressor());
    imgStream.push(io::file_source(imgFname.c_str()));

    uint32_t magic = read32u(imgStream);
    if(magic != 0x803)
    {
      imgStream.reset();
      return false;
    }
    numEntries = read32u(imgStream);
    numRows = read32u(imgStream);
    numCols = read32u(imgStream);

    labStream.reset();
    labStream.push(io::gzip_decompressor());
    labStream.push(io::file_source(labFname.c_str()));

    magic = read32u(labStream);
    if(magic != 0x801)
    {
      imgStream.reset();
      labStream.reset();
      return false;
    }
    int numLabs = read32u(labStream);
    if(numLabs != numEntries)
    {
      imgStream.reset();
      labStream.reset();
      return false;
    }
    return true;
  }

  bool Data::load()
  {
    labStream.reset();

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

    if(!loadImgNLabFile(imgFname, labFname))
      return false;

    cout << "NumEntries:" << numEntries << ", NumRows:" << numRows << ", NumCols:" << numCols << endl;
    dataSize = numRows * numCols;
    unique_ptr<uint8_t> img(new uint8_t[dataSize]);
    imgData = move(img);

    return true;
  }

  // void Data::displayImage()
  // {
  //   Gnuplot plot("image");
  //   plot.plot_image(imgData.get(), numCols, numRows, "MNIST Training Set : 0");
  // }
}
