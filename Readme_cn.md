[English](/Readme.md) | 简体中文

## Introduction

这是一个使用C++来部署Pytorch模型的示例，使用的是libtorch，它是Pytorch的C++版本，可以在不依赖Python环境的情况下使用Pytorch。

代码支持Windows平台和Linux平台，使用的IDE是Clion，编译器是MSVC和GCC。

仓库代码已经在Torch 2.1.0和GPU上测试通过。

## 更新日志

🌟 2023/11/23 发布了 v1.0.0 版本

- 支持Windows平台和Linux平台。
- 支持CPU和GPU。

## 如何在Windows平台上构建可执行文件

### 环境准备

1. 下载并安装Visual Studio, GPU版本需要MSVC编译, CPU版本可以使用MinGW编译
2. [可选]下载并安装MinGW-w64，CPU版本需要MinGW编译
3. [可选]下载并安装CUDA，GPU版本需要CUDA
4. 下载并安装cmake，添加到环境变量
5. 下载并解压libtorch，添加到环境变量
6. [可选]下载并安装Anaconda，添加到环境变量，用于创建配置Python环境，将Torch模型转换为Torch Script模型
7. 下载并安装Windows版本的OpenCV，添加到环境变量，用于读取图片
8. [可选]下载并安装GDAL，添加到环境变量，用于读取遥感影像，推荐使用VCPKG安装
9. 下载并安装Clion，用于编译代码

### 构建可执行文件

1. 下载或克隆仓库代码
2. 使用Clion打开仓库代码
3. 配置Clion的编译器，如果是CPU版本，可以使用MinGW编译，如果是GPU版本，使用MSVC编译
4. 配置Clion的CMake，设置CMake的路径
5. [可选]导出Pytorch模型为Torch Script模型，将模型放在本项目的目录下
6. 修改CMakeLists.txt，设置LibTorch的路径，设置OpenCV的路径，设置GDAL的路径
7. [可选]更改CMakelists.txt，设置CXX_STANDARD的版本
8. [可选]修改main.cpp，设置模型路径，设置图片路径等
9. 使用Clion编译代码，生成可执行文件


## 如何在Linux平台上构建可执行文件

### 环境准备

1. 下载并安装cmake，添加到环境变量
2. 下载并解压libtorch，添加到环境变量
3. [可选]下载并安装Anaconda，添加到环境变量，用于创建配置Python环境，将Torch模型转换为Torch Script模型
4. 下载并安装OpenCV，添加到环境变量，用于读取图片
5. [可选]下载并安装GDAL，添加到环境变量，用于读取遥感影像

### 构建可执行文件

1. 下载或克隆仓库代码
2. [可选]导出Pytorch模型为Torch Script模型，将模型放在本项目的目录下
3. 修改CMakeLists.txt，设置LibTorch的路径，设置OpenCV的路径，设置GDAL的路径
4. [可选]更改CMakelists.txt，设置CXX_STANDARD的版本
5. [可选]修改main.cpp，设置模型路径，设置图片路径等
6. 使用cmake构建项目，make编译项目，生成可执行文件，命令如下：
```bash
mkdir build && cd build
cmake .. -G "Unix Makefiles"
make -j4
```


