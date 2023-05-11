# Deformable mesh simulator

My graduation thesis project. *An Efficient Deformable Object Simulator Based on GPU and Projective Dynamics*.

## Build

The build process of this project is tested under

* Arch Linux x86_64
* gcc 12.2.1
* nvcc 11.8

and should be able to run on Windows or OSX as well.

Generally, use the standard cmake build pipeline:

```
mkdir build
cd build
cmake ..
make
```

And you will need to modify the following line at `/CmakeLists.txt`:

```
set(CMAKE_CUDA_ARCHITECTURES 86) # set __CUDA_ARCH__ to be the latest to generate fastest code (require latest device)
```

if your CUDA architecture version is less than 86.  

And you will need to modify the include directory specification at `/pd/CMakeLists.txt` and `/util/CmakeLists.txt`:

```
include_directories("/opt/cuda/include")
```

if your CUDA header file directory is different.

Also, the building process downloads the latest version of [libigl](https://github.com/libigl/libigl), Eigen and ImGUI, etc. from GitHub, so keep the network running.

## Usage and Demo

Follow the demos below to create yourself a simulation sandbox.

Simulation of a large cloth:

![cloth](./docs/large-cloth.gif)

Apply wind on cloth:

![wind](./docs/wind.gif)

Simulation of Bunny:

![bunny](./docs/bunny.gif)
