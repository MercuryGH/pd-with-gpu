# 问题

![1673774900019](image/P-free_draft/1673774900019.png)

原文：

Computing the matrix product of R is similar to applying a Laplacian-like operator over the neighborhood of a vertex: the first-order A-Jacobi is equivalent to the regular Jacobi iterating vertex’s one-ring neighbors, and the second-order A-Jacobi covers two-ring neighbors of the vertex etc.

Edges on the mesh house the corresponding weights, which are non-zero elements in B.

In our GPU implementation, we

* first precompute all the non-zero A-products offline and load them into the GPU global memory.
* If an A-product is much smaller than others (by two or three orders) due to the disparity of different constraint stiffness, the corresponding computation will be skipped.
* During the simulation run time, we update A-coefficients in parallel after barrier constraints are built, knowing A-coefficients are unchanged for the entire inner solve.

Before an A-Jacobi iteration starts, we need to group adjacent vertices in one CUDA block to leverage their overlapped neighborhoods. A-coefficients of all the vertices within the block are then fetched into local shared memory in parallel to reduce memory latency. The computation at each vertex is further split into several segments so that the parallelism of the computation can be fully scaled to match the performance of the GPU.

问题：

1. 为什么会有“图”，如何建模出图来；为什么k阶A-Jacobi是迭代k邻接结点。
2. 后面的Precomputation
3. 在仔细读Jacobi算法的过程中，我还发现了一些作者笔误（或者是我的理解错误），它们都在

## 复现操作指南

* Cloth, 20 (default), 20 (default), compute
* Constraint, edge length, set wi to 100000.0 (default / 10), set Positional constraint wi to 100000000.0 (default / 10) Active, Apply (#Constriant should be 1121)
* Shift + Left Mouse Click to set 2 pinned points (After that, #Constraint shoud be 1123)
* Picking, 4000.000 (default * 10)
* Physicis, set mass per particle to 1.0 (default / 10), set Gravity and Simulate.

## 单词

* tetrahedron 四面体。
* tetrahedra *tetrahedron*的复数形式。

## 笔记

Local-Global alternation也叫做block coordinate descent。它与牛顿法不同（但或许可以找到……？）

Local step: fix q, minimze p.

Global step: fix p, minimze q. 求导得到一个线性系统，它的系数矩阵通常可以预计算，因此没什么问题。

## 几何笔记

Mesh: 在$R^3$上嵌入的图。

Boundary edge: 仅与一个面邻接的边。

Regular edge: 与恰好两个面邻接的边。

Singular edge: 与多于两个面邻接的边。

Closed mesh: 没有Boundary edge的mesh。

Manifold mesh: 没有Singular edge的mesh。非流形可能会具有一些不好的性质，一般不予考虑。

## Constraint的实现

利用性质internal physical constraints are translation invariant，实现快速收敛

choose $A_i = B_i$ as differential coordinate matrices, 使用Subtracting the mean的方法构造$A_i$。

$n$表示总顶点数，$v_i$表示与约束关联的第一个顶点，$v_j$表示与约束关联的第二个顶点。

### Positional

只与一个顶点相关。简单起见，

$$
A_i = B_i=I_3
$$

为3阶恒等矩阵，行数为$3$，列数为$3n$的选择矩阵

$$
S_i = 
\left[\begin{matrix}
0 & \dots & 1  & \ & & \dots&0\\
0 & \dots & & 1  &  & \dots&0\\
0 & \dots & & & 1 & \dots & 0
\end{matrix}  \right]
$$

的第一行的$1$出现在第$1$行，第$3v_i$列（行列均从零数起，以后记为$(1,3v_i,1)$）。矩阵中只有三个1元素，其它均为0。

可以证明，

$$
(A_iS_i)^TA_iS_i =
\left[\begin{matrix}
0 & \dots \\
\vdots \\
& 1 \\
& & 1 \\
& & & 1 & \dots\\
\vdots \\
\end{matrix}  \right]
$$

为$3n \times 3n$方阵，其中只有三个非零元素$(3v_i,3v_i,1), (3v_i + 1,3v_i + 1,1),(3v_i + 2,3v_i + 2,1)$。

### Edge length

只与两个顶点相关。正确表示行数为$6$，列数为$3n$的选择矩阵$S_i$的方式为

$$
S_i = 
\left[\begin{matrix}
0 & \dots & 1  & \ & & &&&&\dots&0\\
0 & \dots & & 1  &  & &&&&\dots&0\\
0 & \dots & & & 1 & &&&&\dots & 0\\
0 & \dots & & & & \dots& 1  & \ & & \dots&0\\
0 & \dots & & & & \dots&  & 1  &  & \dots&0\\
0 & \dots & & & & \dots& &  & 1 & \dots & 0
\end{matrix}  \right],
$$

其中。第一行的$1$出现在第$3v_i$列，第四行的$1$出现在第$3v_j$列。不一定满足$v_i<v_j$。

而$A_i=B_i$采用Subtracting the mean的方法构造，定义$V$为所有顶点的所在集合，$v_1, \dots,v_n \in \R^3$为顶点位置，$V_i$为约束$i$所包含的顶点集合，$n_i= |V_i|$。那么

$$
A_i = (I_{n_i} - \frac{1}{n_i}1_{n_i}) \otimes I_3
$$

其中$\otimes$为Kronecker product，$1_{n_i}$是$n_i$阶全1方阵。对于Edge length本例，$n_i=2$，于是

$$
A_i = (I_2 - \frac{1}{2}1_2) \otimes I_3=
\left[\begin{matrix}
\frac{1}{2}I_3 & -\frac{1}{2}I_3 \\
-\frac{1}{2}I_3 & \frac{1}{2}I_3
\end{matrix}  \right].
$$

为一个$6 \times 6$矩阵。

故$(A_iS_i)^TA_iS_i$为一个$3n \times 3n$方阵，其中只有$4$个$3 \times 3$块的对角元素非$0$，即只有$12$个元素非0。可以取$v_i=2, v_j=3$自行找规律计算。

# 讲稿

大家好，我是陈瀚洋，接下来由我来分享一个在物理仿真领域发表在 2022SigGraph上最新的工作，Penetration-free
Projective Dynamics on the GPU.

这个工作的核心是，在利用连续碰撞检测（CCD）保证不穿透的前提下，为可变形体提供一个高效的仿真方法

---

一般可变形体的运动规律都可以由牛顿第二定律直接描述。对于计算机而言，我们将这个微分方程转换成时间离散的差分方程。

求解这样的差分方程可以采用隐式欧拉法，或者把它转换为等价的优化问题，使用牛顿法求解，但原始的数学模型求解成本很高。因此诸如一些经验性的方法，以及Position
Based Dynamics, Projective Dynamics等近似方法不断涌现，PD是2014年提出的一个，就是其中应用相对较广的一个方法。

---

PD将原来的非线性优化中的势能项用一个比较理想的二次项代替，同时将整个复杂的问题分解成若干个独立的子问题，每个子问题都是一个关于少量顶点的一个约束问题，因此在牺牲一定精度的条件下换取更快的计算效率。

每个子问题的求解目标都是当前顶点的投影位置，每一步都会把顶点投影到一定的位置来满足约束，这可以用Local-Global交替迭代的方式求解，其中Local这一步是天生适合并行计算的，Global
Step是一个可以进行系数矩阵预计算的线性方程组。

---

论文中用到的连续碰撞检测方法是IPC，这是一项SigGraph 2020的工作。我们知道连续碰撞检测要求点-三角形 和
边-边
各碰撞检测元之间定义的距离不小于0，以避免穿透，这样一来就是一个带约束的优化问题，是很难求解的。IPC将这个约束用Barrier
Function表示出来，提供了一个非线性弹簧的效果。

可以看出来Barrier Function是带参数的，当两个碰撞检测元之间的距离小于这个Barrier的时候，就会迅速产生很大的惩罚，将两个物体推开，达到碰撞避免的效果。

---

事实上，处于效率考虑，一般的PD方法都是与离散碰撞检测（也就是DCD）结合起来的，这会导致可能出现的穿模现象。

那么直接把PD和CCD结合起来会怎么样呢？论文中探讨了这一想法，并得出结论，PD和CCD直接结合会导致一些问题，需要针对性地设计解决方案。

这是因为，PD的算法中是没有内力这一概念的，投影位置会成为问题。

看例子，一个点初始在障碍我的左边，经过global
step后计算位置发现它穿过了障碍物，然后我们进行CCD将其推到安全位置。

但是并不会改变物体原有的惯性，因此在下一个local-global迭代中，点仍然会试图穿过障碍物，碰撞检测还是会进行，以此循环往复，得到的效果就像这个动图一样，兔子好像在碰撞后就“粘在了地上”。这种效果是不好的。

那么我们再换另一种方式，用IPC改良的CCD的效果如何呢？这里我们需要预先设置参数d hat，保证d hat之内的点受到位置约束。但由于d hat本身很小，不能提供一个有效的把弹性体弹开的冲量。这样做本质上并没有改变什么，效果与一般方法是一致的。

（根本原因是PD中没有表达内力）

---

为此，论文中提到的方法就是基于冲量守恒，表示一个barrier
projection表示回弹。

这个想法其实也比较自然，就是只要CCD被激活了，就根据这个公式进行回弹位置的计算。最后的结果也是比较自然的。

---

那么现在我们解决了PD+CCD中碰撞约束的问题，但是作者在实验过程中还是出现了两者结合中迭代不收敛的问题。

这张图中绿色的曲线反映了在PD过程中一直在发生剧烈的抖动，这说明碰撞检测和修复会把已经优化好的低势能点重新放到高势能点上。

这个问题的根源是我们每次进行Local
Global CCD这种单纯的迭代方式中，CCD都可能导致位置的突变，从而导致势能突变。

这样频繁的调用CCD可能导致正常的Local Global迭代收敛缓慢，甚至进入死循环，这张图就反映了这个问题。

发现这个问题后，作者将单层迭代改成了双层迭代的方式，在内层循环尽量先求解Local
Global PD，跳出内层循环，回到外层循环后再调用CCD。这样就避免了CCD的频繁调用，从而有效地缓解了这种问题的出现。

但是这种方法需要更多的迭代次数，计算效率不高，而作者认为local
step已经没有多少加速的空间，因此他们探索了global
step的加速空间，并设计出了一种并行算法。

---

A-Jacobi算法就是作者提出的加速global step的并行算法。Jacobi就是大家熟悉的数值分析里的一种解线性方程组的迭代法，系数矩阵可以写成两个矩阵之差，其中D是对角矩阵，B矩阵是非对角矩阵。

那么，我们来分析一下在并行计算中每个线程处理的工作是什么。可以看到，如果我们理想情况下对每个顶点分配一个线程，每个线程只需要执行几个乘法和加法操作，对于今天的GPU而言还是很轻量化的，不能拉满计算能力。作者就想到了把Jacobi的计算步子迈大一些，让GPU每次都多算几步，避免CPU和GPU之间的一些调度代价。

这里就是高阶Jacobi并行计算的思路。

---

在作者的这种迈大步地思路下，可以推导得到A-Jacobi的一般形式，这个式子的l表示的就是Jacobi的阶数。

那么我们来看一下这个式子的计算过程是怎么样的。可以看到这里出现了一个高次的R矩阵乘以一个向量的计算。我们怎样去快速实现呢？

作者观察到在一阶的情况下，R矩阵乘上一个向量的本质可以看成是遍历系数矩阵中元素的邻接元素。二阶情况也是如此，只不过邻接边数变成了2。

注意到当不发生碰撞的情况下，系数矩阵是常量，所有的资源都可以进行预计算；论文又证明了在发生碰撞之后，只有系数矩阵的对角元素发生改变，非对角元素不会发生改变，因此B矩阵是始终可以预计算的。具体的实现细节paper中有所介绍，这里就不展开了。

---

这里的横坐标表示未知数的规模，纵坐标表示加速的效率。其中1阶Jacobi是baseline，都是100%。二阶和三阶Jacobi的效率分别用黄色和绿色表示出来。

从图中可以看出，在规模不大的情况下，A-Jacobi的阶数越高，效率越高，但随着系统规模越大，加速的效果就逐渐抹平了。这是因为随着规模的提高，GPU的计算资源也逐渐饱和了。

---

这里是作者跟其他几个线性系统的solver做的对比，可以看到在使用切比雪夫加速后，A-Jacobi算法的性能比较接近PCG算法，但A-Jacobi在GPU上实现，PCG对GPU不友好，因此作者认为他们的方法更具泛用性。

---

作者在一个YouTube视频中展示了论文中方法结合起来的效果，整个系统有15万个自由度，论文的方法与IPC相比并没有效果上的不同，但效率是IPC的两千倍。

---

总之，作者的算法为可变形体的物理模拟提出了一个高效的解决方案。我的分享到此结束，谢谢大家。
