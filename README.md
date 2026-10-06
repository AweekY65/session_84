# sparse_linalg — 本地稀疏线性代数模块

纯本地、header-only 的 C++17 稀疏线性代数库。所有矩阵、向量、求解状态与
测试数据仅存在于本地内存中，不依赖 MATLAB、远程计算服务或外部数据库。
CSR 存储与 CG 求解器均为自行实现；测试中的稠密参考计算也在本地完成
（可选用 Eigen 作为对照，但非必需）。

## 目录结构

```
include/sparse_linalg/csr_matrix.hpp   # CSR 稀疏矩阵 + triplet 构建
include/sparse_linalg/cg_solver.hpp    # 预条件共轭梯度求解器
tests/test_sparse_linalg.cpp           # 终端自动化测试
CMakeLists.txt
```

## CSR 存储布局

`CsrMatrix` 使用标准 Compressed Sparse Row 布局：

- `values[k]`：第 k 个非零元的值；
- `col_idx[k]`：第 k 个非零元的列索引；
- `row_ptr[i]`：第 i 行非零元在 `values`/`col_idx` 中的起始位置，
  行 i 占据区间 `[row_ptr[i], row_ptr[i+1])`，`row_ptr` 长度为 `rows + 1`。

通过 `CsrMatrix::from_triplets(rows, cols, triplets)` 从 COO triplet 数据
构建：先按 `(row, col)` 排序，相同坐标的重复元素**求和合并**（允许相消为
显式零），并保证每行内列索引**升序**。越界坐标抛出 `std::out_of_range`。

`matvec(x)` 实现稀疏矩阵-向量乘 `y = A x`，仅遍历非零元，维度不符时抛出
`std::invalid_argument`。

## Conjugate Gradient 求解器

`cg_solve(A, b, options)` 求解对称正定（SPD）系统 `A x = b`：

- 标准预条件 CG 迭代，初始猜测 `x0 = 0`；
- **Jacobi 预条件**：`M = diag(A)`，通过
  `CgOptions::use_jacobi_preconditioner` 开关（默认开启）；
- **收敛判据**：相对残差 `||r_k||_2 / ||b||_2 <= tolerance`
  （`CgOptions::tolerance`，默认 1e-8）；`b = 0` 时直接返回精确解 `x = 0`；
- `CgOptions::max_iterations` 限制迭代预算（默认 1000）。

求解过程只调用 `matvec`，**绝不**把稀疏矩阵转换为稠密矩阵。

`CgResult` 报告：`status`、`converged`、`iterations`（实际迭代次数）、
`final_residual`（最终相对残差）与 `solution`。

### 状态码

| `CgStatus` | 含义 |
|---|---|
| `Converged` | 达到容限 |
| `MaxIterationsReached` | 迭代预算耗尽 |
| `NotSquare` | 矩阵非方阵 |
| `DimensionMismatch` | 右端项维度不符 |
| `NotPositiveDefinite` | 检测到非 SPD（对角元 ≤ 0、`pᵀAp ≤ 0` 或 `rᵀz ≤ 0`），零矩阵也在此类 |
| `NumericalError` | 迭代中出现 NaN / Inf |

## 构建与测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/test_sparse_linalg     # 或: ctest --test-dir build --output-on-failure
```

测试在终端输出每项检查结果与求解残差，覆盖：

- CSR 布局（含空行、列索引有序）；
- triplet 重复坐标合并（含相消为零）；
- SpMV 与稠密参考结果一致；
- 已知精确解的 1D Poisson 系统；
- Jacobi 预条件的迭代次数对比（159 → 80）；
- 零矩阵、非方阵、维度错误、不定矩阵、NaN 输入的状态报告；
- 随机稀疏 SPD 系统（`A = MᵀM + nI`，n=200）；
- 大规模 2D Poisson 系统（90,000 未知数，448,800 非零元，纯 CSR 求解，
  并用一次额外 SpMV 独立验证残差）。
