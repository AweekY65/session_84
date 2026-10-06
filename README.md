# Local Sparse Linear Algebra Module

一个完全本地化的稀疏线性代数模块：所有矩阵、向量、求解状态和测试数据只存在于
本地文件或内存中，不依赖 MATLAB、远程计算服务或外部数据库。CSR 存储与共轭梯度
（CG）求解算法均为自行实现；Eigen 仅在测试中作为参考解对照。

## 目录结构

```
include/sparse/csr_matrix.hpp   # CSR 稀疏矩阵（triplet 构建、spmv）
include/sparse/cg.hpp           # CG 求解器（Jacobi 预条件、状态报告）
tests/test_main.cpp             # 终端自动化测试（打印残差）
CMakeLists.txt
```

## CSR 存储布局

`CsrMatrix` 使用三个数组表示一个 `rows x cols` 的稀疏矩阵：

- `values[k]`：第 k 个非零元的值；
- `col_idx[k]`：第 k 个非零元的列索引；
- `row_ptr[i]`：第 i 行在 `values`/`col_idx` 中的起始偏移，
  第 i 行占据区间 `[row_ptr[i], row_ptr[i+1])`。

通过 `CsrMatrix::fromTriplets(rows, cols, triplets)` 从坐标（triplet）数据构建：

1. 同一 `(row, col)` 坐标上的重复元素按求和合并；
2. 合并后恰好为零的元素被丢弃；
3. 每行内部的列索引按升序排列（二分/对角查找可利用该有序性）。

越界坐标抛出 `std::out_of_range`。

## 稀疏矩阵-向量乘法（SpMV）

`A.spmv(x)` 返回 `y = A * x`，仅按 CSR 行区间做乘加，时间复杂度 `O(nnz)`，
测试中与稠密参考计算逐元素比对（误差为 0）。

## Conjugate Gradient 求解器

`solveCG(A, b, options, x0)` 求解对称正定（SPD）系统 `A x = b`：

- `CgOptions::tolerance`：相对残差容限（默认 1e-10）；
- `CgOptions::maxIterations`：最大迭代次数（默认 1000）；
- `CgOptions::useJacobiPreconditioner`：是否启用 Jacobi（对角）预条件（默认开）。

**收敛判据**：`||r_k||_2 <= tolerance * ||b||_2`，其中 `r_k = b - A x_k`。

**Jacobi 预条件**：取 `M = diag(A)`，每次迭代以 `z = M^{-1} r` 替代裸残差进入
搜索方向更新（标准 PCG 流程）。SPD 矩阵对角元必严格为正，因此构建预条件时
同时完成了"明显非正定"筛查。

**结果报告**（`CgResult`）：`status`、`iterations`（迭代次数）、
`finalResidual`（终止时绝对残差范数）、`solution`、`converged()`。

**状态码**（`CgStatus`）：

| 状态 | 含义 |
|---|---|
| `Converged` | 达到容限 |
| `MaxIterationsReached` | 迭代预算耗尽 |
| `NotSquare` | 矩阵非方阵 |
| `DimensionMismatch` | 右端项/初值维度不符 |
| `NotPositiveDefinite` | 对角元非正（明显非正定，含零矩阵） |
| `NumericalBreakdown` | 输入含 NaN/Inf，或迭代中出现 `p'Ap <= 0` / 非有限值 |

**内存保证**：求解全程只操作 CSR 数组与若干长度为 n 的向量，
绝不把大规模稀疏输入转换为完整稠密矩阵。

## 构建与测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/sparse_tests        # 直接运行，逐条打印残差
ctest --test-dir build --output-on-failure   # 或通过 CTest 运行
```

若 Eigen 不在系统默认路径，可用 `-DEIGEN3_INCLUDE_DIR=/path/to/eigen3` 指定
（Eigen 仅用于测试参考）。

测试覆盖：CSR 构建与布局、重复元素合并与列排序、SpMV 对稠密参考、
已知精确解的 Poisson 系统、Jacobi 预条件加速效果、零矩阵、非正定输入
（负对角元与不定矩阵）、非方阵/维度错误/NaN 输入，以及 2000 阶随机稀疏
SPD 系统（与 Eigen 的 CG 参考解对照，输出残差与解差）。
