# README

PD testing.

## Problems with build

* 使用 cmake 项目时，VS 不识别 cuda_runtime.h

这是 VS 的问题，需要在相应的 `CMakeLists.txt` 显式地加入 `include_directories("E:\\CUDA\\v11.6\\include")`。

* 使用 libigl (ver <= 2.3.0) 时，nvcc 报错 `single file input`

这是因为那个让 nvcc 进行编译的命令有问题，而这个编译命令的产生源头是 libigl 的某个 cmake文件。具体而言，`/FS /bigobj` 对于 nvcc 而言不可理解，它们就被解析成了“文件名”，并且找不到这些文件，因此报一个实际与文件完全无关的 `single file input` 错误。解决方案有：

1. 将 libigl 升级到解决了该 bug 的版本 (ver >= 2.4.0)。
2. 手动删除 nvcc 编译命令中的 `/FS /bigobj` 语句，更好地，这些语句的正确位置应该是 `-Xcompiler=""` 中的引号内部的位置，将它们移动到其中即可。这可以通过修改 libigl 中的 cmake 文件；或者，在构建出来的 VS 工程中，选择 VS 项目资源管理器 -> pd 右键-属性 -> Cuda C++ -> 命令行 -> 其他选项 中，直接修改实际构建语句即可。

值得注意的是，libigl ver >= 2.4.0 的 cmake 构建方式与之前的版本有很大不同，官方文档和样例工程 libigl-example-project 可能并没有全面更新，需要参考 https://github.com/libigl/libigl/releases/tag/v2.4.0 获取实际构建方式。

## 当前存在的Bugs

* constraints 数目过多时，预计算local step GPU 会崩溃。最大数目为42000 constraints。可用120\*120和130\*130布料测试。

> 原因： https://stackoverflow.com/questions/70024184/cuda-complex-object-initialization-within-device-problem-with-cudadevicesetlimi

* bunny_s, 50*50布料 等模型中A-Jacobi算法出现错误，部分顶点的迭代值强制收敛到0，可能是GPU memory access UB导致的。

> 是的。具体的原因是调用kernel时 n_vertex 参数传入错误导致了边角料问题。该问题在边角料很多的时候会导致memory access UB error，但不多时不会导致error，只会在计算上出现错误。

## CMake 指定编译生成类型

```
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake .. -DCMAKE_BUILD_TYPE=Release
```

## CUDA 概念

一个 kernel 对应一个 grid，kernel 是 CPU 的占位符，grid 在 GPU 上是函数的实际运行时。kernel 编号不一定与 grid 编号相同。

kernel calling format: `<<<#Blocks in a grid, #Threads in a block>>>`

blocks in a grid 和 threads in a block 在逻辑上都可以是三维的，但在实际上不一定。

#Blocks in a grid 和 #Threads in a block 的最大值都可以在 deviceQuery 中看到，类似这样：

```
Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
```

sm: streaming multiprocessor，GPU大核。
warp: 线程束
lane: warp 里面的任意一个 thread

## cuda-gdb 注意事项与常用指令

* 单步操作会同时在一个 warp 上的所有 lane 上生效（特殊情况：`__syncthread`）

```
cuda-gdb ./pd/debug_pd

b gpu_local_step
b *0x00007fffc7284290

```

## Profiling

### 测试环境

* OS: Arch Linux x86_64
* CPU: 12th Gen Intel(R) Core(TM) i7-12700
* GPU: NVIDIA Corporation GA104 [GeForce RTX 3060 Ti Lite Hash Rate]

<!-- * C++ Compiler: g++ version 12.2.1 20230201 (GCC) gcc版本高于cuda时，可能会让cmake cuda无法完成编译 -->

* CUDA Compiler: Cuda compilation tools, release ??.? (会随Arch Linux pacman更新，要小心)

### 坑点

不同版本的CUDA有不同的编译器版本上限，例如cuda 11.1就不支持gcc 12+，所以要么升级cuda，要么降级gcc。

### 测试结果

100 * 100、90 * 90 布料、bunny_l，A-Jacobi-1, 2, 3 (Itr Solver #Itr = 700, PD #Itr = 2) 都可以胜过 Direct

当PD #Itr增大时，A-Jacobi变慢的速率比Direct快，所以完全不占优势。

#### 测试结果与顶点数的关系

当 n_vertex 较小时，两者基本没有差距（Direct略胜一筹）。

当 n_vertex 较大时，Direct虽然不快，但A-Jacobi由于需要收敛，所以迭代次数更多，实际更慢，因此Direct胜一筹。

（事实上都不一定）

## 扩展计划

### viewer扩展到多物体，Mesh扩展到Deformable Mesh和Static Mesh

* 一个 `viewer.data_list[idx]`中的对象绑定一个Imguizmo widget，运行时通过一个哈希表 `<idx, Matrix>`更新其变换矩阵。

## Random Error

GL_INVALID_VALUE，call stack 为

```
#7  0x00007ffff7a6854a in fprintf () from /usr/lib/libc.so.6
#8  0x00005555555b3837 in igl::opengl::report_gl_error(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) ()
#9  0x00005555555b389e in igl::opengl::report_gl_error() ()
#10 0x00005555555b396b in igl::opengl::destroy_shader_program(unsigned int) ()
#11 0x00005555555b6005 in igl::opengl::MeshGL::free()::{lambda(unsigned int&)#1}::operator()(unsigned int&) const ()
#12 0x00005555555b604f in igl::opengl::MeshGL::free() ()
#13 0x00005555557228bd in igl::opengl::glfw::Viewer::erase_mesh(unsigned long) ()
```

有概率在remove mesh时触发，表现为stdout中输出 `invalid value`，对应用似乎无影响。

## constriant建模

布料使用全边弹簧（必要），全顶点bending（不必要）

模拟旗子被撕裂：当风力过大时，移除edge spring constraint（又叫edge strain constraint）

体积物体使用volumetric strain constraint，包括但不限于volumne preservation, strain constraint. 必须使用tetrahedral mesh.
