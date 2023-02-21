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

constraints 数目过多时，预计算local step GPU 会崩溃。最大数目为42000 constraints 

