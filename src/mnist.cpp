#include <stdint.h>
#include <iomanip>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include "mnist.hpp"
#include "gnuplot_i.hpp"

using namespace std;
using boost::filesystem::path;
namespace io = boost::iostreams;

#define htonl(x) (((x)<<24) | (((x)<<8) & 0xFF0000) | (((x)>>8) & 0xFF00) | ((x)>>24))

uint32_t read32u(io::filtering_istream& inf)
{
    uint32_t tmp = 1234;
    inf.read((char *)&tmp, sizeof(tmp));
    return htonl(tmp);
}

namespace mnist
{
  bool Data::load()
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

    io::filtering_istream imgStream;
    imgStream.push(io::gzip_decompressor());
    imgStream.push(io::file_source(imgFname.c_str()));

    uint32_t magic = read32u(imgStream);
    if(magic != 0x803)
      return false;
    uint32_t numImgs = read32u(imgStream);
    uint32_t numRows = read32u(imgStream);
    uint32_t numCols = read32u(imgStream);
    cout << "NumIngs:" << numImgs << ", NumRows:" << numRows << ", NumCols:" << numCols << endl;
    uint8_t * img = new uint8_t[numCols * numRows];
    imgStream.read((char *)img, numCols * numRows);

    Gnuplot plot("image");
    plot.plot_image(img, numCols, numRows, "MNIST Training Set : 0");
    sleep(10);

    return true;
  }
}
