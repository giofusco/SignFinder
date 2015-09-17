# README #

For Windows installation instructions see "doc/Instructions_Wicab Sign Finder.pdf"

Dependencies
=============
SignFinder depends on two external projects, LibSVM and OpenCV.

1. [OpenCV](http://www.opencv.org)
-------------------------------
Usually the version of OpenCV available at package manager repositories is outdated. We are currently using OpenCV 2.4.11. The best way is to download the opencv source code and compile it from scratch.

Download OpenCV( [Windows](https://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.11/opencv-2.4.11.exe/download) / [Linux/OSX](https://github.com/Itseez/opencv/archive/2.4.11.zip)) Installation instructions can be found [here](http://docs.opencv.org/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html)


2. [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
----------------------------------------------------
For *nix / OSX you can download and install libsvm via a package manager, such as synaptic for Ubuntu or 
macports for OSX. Make sure you install the headers as well as the library.

Alternatively you can download it as a [single zip file](http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+zip). Just unzip the folder and copy the svm.h and svm.cpp files to the src/ folder in this repo. You can also use the libsvm [git repository](https://github.com/cjlin1/libsvm) to get these files.


Building SignFinder
=======================
To build, use [cmake](http://www.cmake.org/).

To configure the build, create a `build/` directory under the source root folder, and change to this folder

    >> mkdir build
    >> cd build
    >> cmake ..

If cmake cannot find your existing LibSVM installation, you can provide the folder where the installed headers and libraries are by defining the `LIBSVM_ROOT_DIR` variable, i.e. if the headers and libraries are in `/usr/local`, 

    >> cmake -DLIBSVM_ROOT_DIR=/usr/local/ ..

You can then compile and install the project using make

    >> make
    >> make install

If building for OSX, you can generate Xcode projects using the `-GXcode` flag. These can then be build using Xcode or from the command line using the `xcodebuild` command, i.e., instead of make, use

    >> xcodebuild -project SignFinder.xcodeproj

If building for Windows, you can use [mingw](http://sourceforge.net/projects/tdm-gcc/files/TDM-GCC%20Installer/tdm64-gcc-4.9.2-3.exe/download) and mingw-make. Alternatively, cmake can generate a Visual Studio project, see [here](http://www.cmake.org/cmake/help/v3.0/manual/cmake-generators.7.html) for the cmake options to generate a project for your particular version of Visual Studio.

Documentation
=====
In addition to the build instructions in this file, you can find a tech report providing an overview of the algorithms in the build/doc folder after running cmake.

Building API documentation
--------------------------

The API documentation requires [Doxygen](http://www.stack.nl/~dimitri/doxygen/index.html) to be installed. This can be generated at build time via passing cmake the `BUILD_DOCUMENTATION` flag, e.g.,

    >> cmake -DBUILD_DOCUMENTATION=ON ..
    
This will create a `doc` make target, and is ON by default. This also provides a build target `doc` such that documentation can also be built by 

    >> make doc
    
The resulting html documentation will be put in `build/doc/html`. Just open the index.html file in any web browser.

Alternatively, you can build it yourself using doxygen later using the doxygen configuration file `Doxyfile` in `build/doc`. While in `build`,

    >> cd doc
    >> doxygen Doxyfile
    
Running SignFinder
===================

```
USAGE: SignFinder -c configfile [-p prefix] [-m maxdim] [-s] [-d] [-f] [-t] [-n] [-o output] input
  -1                          input. Either a file name, or a digit indicating webcam id                            
  -c, --configFile            location of config file                         
  -d, --debug=[false]         whether to show intermediate detection stage results                                         
  -f, --flip=[false]          whether to flip the input image                 
  -h, --help=[true]           print this message                              
  -m, --maxdim=[640]          maximum dimension of the image to use while processing                              
  -n, --notrack=[false]       whether to turn off tracking                    
  -o, --output                if a name is specified, the detection results are saved to a video file given here
  -p, --patchPrefix           prefix for dumping detected patches to disk if one is provided                         
  -s, --saveFrames=[false]    whether to save frames                          
  -t, --transpose=[false]     whether to transpose the input image            
  -v, --version=[false]       version info                                    
```
