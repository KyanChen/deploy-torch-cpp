English | [简体中文](/Readme_cn.md)

## Introduction

This is an example of using C++ to deploy Pytorch models. It uses libtorch, which is the C++ version of Pytorch and can be used without relying on the Python environment.

The code supports Windows platform and Linux platform, the IDE is Clion, and the compiler is MSVC and GCC.

The repository code has been tested on Torch 2.1.0 and GPU.

## Update log

🌟 2023/11/23 Released v1.0.0 version

- Support Windows platform and Linux platform.
- Support CPU and GPU.

## How to build executable files on Windows platform

### Environment preparation

1. Download and install Visual Studio, GPU version needs MSVC compilation, CPU version can use MinGW compilation
2. [Optional] Download and install MinGW-w64, CPU version needs MinGW compilation
3. [Optional] Download and install CUDA, GPU version needs CUDA
4. Download and install cmake, add to environment variables
5. Download and unzip libtorch, add to environment variables
6. [Optional] Download and install Anaconda, add to environment variables, used to create and configure Python environment, convert Torch model to Torch Script model
7. Download and install Windows version of OpenCV, add to environment variables, used to read pictures
8. [Optional] Download and install GDAL, add to environment variables, used to read remote sensing images, it is recommended to install with VCPKG
9. Download and install Clion for compiling code

### Build executable file

1. Download or clone the repository code
2. Use Clion to open the repository code
3. Configure the compiler of Clion. If it is a CPU version, you can use MinGW to compile. If it is a GPU version, use MSVC to compile
4. Configure Clion's CMake and set the path of CMake
5. [Optional] Export the Pytorch model to the Torch Script model and put the model in the directory of this project
6. Modify CMakeLists.txt, set the path of LibTorch, set the path of OpenCV, set the path of GDAL
7. [Optional] Change CMakelists.txt, set the version of CXX_STANDARD
8. [Optional] Modify main.cpp, set the model path, set the picture path, etc.
9. Use Clion to compile the code and generate the executable file


## How to build executable files on Linux platform

### Environment preparation

1. Download and install cmake, add to environment variables
2. Download and unzip libtorch, add to environment variables
3. [Optional] Download and install Anaconda, add to environment variables, used to create and configure Python environment, convert Torch model to Torch Script model
4. Download and install OpenCV, add to environment variables, used to read pictures
5. [Optional] Download and install GDAL, add to environment variables, used to read remote sensing images

### Build executable file

1. Download or clone the repository code
2. [Optional] Export the Pytorch model to the Torch Script model and put the model in the directory of this project
3. Modify CMakeLists.txt, set the path of LibTorch, set the path of OpenCV, set the path of GDAL
4. [Optional] Change CMakelists.txt, set the version of CXX_STANDARD
5. [Optional] Modify main.cpp, set the model path, set the picture path, etc.
6. Use cmake to build the project, make to compile the project, and generate the executable file. The command is as follows:
```bash
mkdir build && cd build
cmake .. -G "Unix Makefiles"
make -j4
```
