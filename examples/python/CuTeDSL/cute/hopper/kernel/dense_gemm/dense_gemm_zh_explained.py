# -*- coding: utf-8 -*-
# 中文块级注释版，原始文件：dense_gemm.py
# 普通 Hopper dense GEMM：host 端准备 tensor/JIT/benchmark，device 端用 TMA + WGMMA 完成 C = A @ B。
# 这是非 persistent 版本：每个 CTA/cluster 主要处理 grid 坐标对应的一个输出 tile。
# 注释风格：只在函数/类、控制流、重要 API 调用和多行逻辑块前解释一次；参数行保持原代码形状。

# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
from typing import Tuple, Type
import math
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils

"""
A high-performance batched dense GEMM (C = A * B) example for the NVIDIA Hopper architecture
using CuTe DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Hopper's WGMMA for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Supports multi-stage pipeline to overlap computation and memory access

This GEMM works as follows:
1. Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. Perform matrix multiply-accumulate (MMA) operations using WGMMA instruction.
3. Store results from registers (RMEM) to shared memory (SMEM), then to global memory (GMEM) with TMA operations.

Hopper WGMMA instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Perform MMA operation and store the result in Accumulator(register)

To run this example:

.. code-block:: bash

    python examples/hopper/dense_gemm.py                                   \
      --mnkl 8192,8192,8192,1 --tile_shape_mn 128,256                      \
      --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
      --c_dtype Float16 --acc_dtype Float32                                \
      --a_major k --b_major k --c_major n

The above example command compute batched gemm with M=8192, N=8192, K=8192,
batch_count=1. The Hopper WGMMA tile shape is 128x256x64 and the cluster shape
is (1,1). The input, mma accumulator and output data type are set as fp16, fp32
and fp16, respectively.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/hopper/dense_gemm.py                               \
      --mnkl 8192,8192,8192,1 --tile_shape_mn 128,256                      \
      --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
      --c_dtype Float16 --acc_dtype Float32                                \
      --a_major k --b_major k --c_major n

Constraints:
* Supported input data types: fp16, fp8 (e4m3fn, e5m2), int8, uint8
* For fp16 types, A and B must have the same data type
* For fp8 types, A and B can have different types (e4m3fn or e5m2)
* For 8-bit integer types, A and B can have different types (int8 or uint8)
* 8-bit types (e4m3fn, e5m2, int8, uint8) only support k-major layout
* CTA tile shape M must be 64/128
* CTA tile shape N must be 64/128/256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 4
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 8, 16 for Float16, and Float8, respectively.

中文说明：
这是一个面向 NVIDIA Hopper 架构、使用 CuTe DSL 编写的高性能 batched dense GEMM 示例，计算 C = A * B。
- 矩阵 A 的逻辑形状是 MxKxL，L 是 batch 维；A 可以是 row-major("K") 或 column-major("M")。
- 矩阵 B 的逻辑形状是 NxKxL，L 是 batch 维；B 可以是 row-major("N") 或 column-major("K")。
- 矩阵 C 的逻辑形状是 MxNxL，L 是 batch 维；C 可以是 row-major("N") 或 column-major("M")。

这个 GEMM kernel 支持：
    - 使用 TMA 高效完成 global memory 和 shared memory 之间的数据搬运
    - 使用 Hopper WGMMA 指令执行矩阵乘加
    - 通过 cluster 上的 TMA multicast 减少 L2/global memory 流量
    - 使用多 stage pipeline 重叠数据搬运和计算

执行流程：
1. 使用 TMA 将 A/B 从 GMEM 搬到 SMEM。
2. 使用 WGMMA 做矩阵乘加，结果累加到寄存器 accumulator。
3. 将寄存器中的结果写到 SMEM，再通过 TMA store 写回 GMEM。

Hopper WGMMA 的核心行为：
- 从 SMEM 读取矩阵 A
- 从 SMEM 读取矩阵 B
- 执行 MMA，并把结果写入 accumulator register

运行示例：上面的命令会计算 M=8192、N=8192、K=8192、batch_count=1 的 batched GEMM。
Hopper WGMMA tile shape 是 128x256x64，cluster shape 是 (1,1)。输入、累加器和输出 dtype
分别是 fp16、fp32 和 fp16。也可以用 NCU profiler 收集性能数据。

约束：
* 支持的输入 dtype：fp16、fp8(e4m3fn/e5m2)、int8、uint8。
* fp16 输入要求 A/B dtype 相同；fp8 和 8-bit integer 输入允许 A/B 使用同宽的不同 dtype。
* 8-bit ((e4m3fn, e5m2, int8, uint8))类型只支持 k-major layout。
* CTA tile M 只能是 64/128，CTA tile N 只能是 64/128/256。
* Cluster shape 的 M/N 必须是正数且为 2 的幂，总 cluster size 不超过 4。
* A/B/C tensor 的连续维至少需要 16 字节对齐。bf16 则num_elements是8的倍数，fp8则num elements是16的倍数
- TMA bulk tensor copy 对连续内存维度通常希望按 16B 粒度对齐/整除


疑问： 在C:\Users\xinji1\Desktop\interv\cutlass\examples\python\CuTeDSL里，有没有cluster size >1 的kernel？
"""


# /////////////////////////////////////////////////////////////////////////////
#  Helpers to parse args
#  参数解析辅助函数
# /////////////////////////////////////////////////////////////////////////////
# 函数 parse_comma_separated_ints：解析逗号分隔的整数列表，例如把 "128,256" 转成 (128, 256)，供 argparse 处理 tile/shape 参数。
# 参数：s；返回：未显式标注。
def parse_comma_separated_ints(s: str):
    # 用 try/except 捕获解析或运行时错误，并转换成更清晰的用户提示。
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    # 异常处理分支：捕获指定错误并执行清理、转换或重新抛出。
    except ValueError:
        # 遇到不支持的 dtype/layout/alignment 或运行环境时主动报错，避免继续生成非法 kernel。
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        )


# 函数 parse_arguments：定义并解析命令行参数，包括问题规模、tile/cluster 形状、dtype/layout、校验和 benchmark 选项。
# 参数：无；返回：argparse.Namespace。
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example of MxNxKxL GEMM on Hopper.")

    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(4096, 4096, 4096, 1),
        help="mnkl dimensions (comma-separated)",
    )
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument(
        "--tile_shape_mn",
        type=parse_comma_separated_ints,
        choices=[(128, 128), (128, 256), (128, 64), (64, 64)],
        default=(128, 128),
        help="Cta tile shape (comma-separated)",
    )
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        choices=[(1, 1), (2, 1), (1, 2), (2, 2)],
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument(
        "--a_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
    )
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument(
        "--b_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
    )
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument(
        "--c_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
    )
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
    )
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument(
        "--warmup_iterations", type=int, default=0, help="Warmup iterations"
    )
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the kernel",
    )
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="Skip reference checking"
    )
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")
    if len(args.tile_shape_mn) != 2:
        parser.error("--tile_shape_mn must contain exactly 2 values")
    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    return args


# /////////////////////////////////////////////////////////////////////////////
#  Host setup and device kernel launch
#  Host 端设置与 device kernel 启动
# /////////////////////////////////////////////////////////////////////////////


# 类 HopperWgmmaGemmKernel：普通 Hopper WGMMA GEMM 的 host/kernel 封装类，保存配置并负责创建 TMA、SMEM layout 与 kernel
# launch。
class HopperWgmmaGemmKernel:
    """
    This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Hopper GPUs.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param tile_shape_mn: Shape of the CTA tile (M,N)
    :type tile_shape_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: Supported A/B data types:
        - Float16
          A and B must have the same data type
        - Float8E4M3FN/Float8E5M2
          A and B can have different types (Float8E4M3FN/Float8E5M2)
          only support k-major layout
        - Int8/Uint8
          A and B can have different types (Int8/Uint8)
          only support k-major layout

    :note: Supported accumulation types:
        - Float32/Float16 (for all floating point inputs)
        - Int32 (for Int8/Uint8 inputs)

    :note: Constraints:
        - CTA tile M must be 64/128
        - CTA tile N must be 64/128/256
        - CTA tile K must be 64
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 4

    Example:
        >>> gemm = HopperWgmmaGemmKernel(
        ...     acc_dtype=cutlass.Float32,
        ...     tile_shape_mn=(128, 256),
        ...     cluster_shape_mn=(1, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, stream)

    中文说明：
    这个类封装 Hopper batched GEMM kernel，支持多种输入 dtype，并使用 Hopper 特有的 TMA、WGMMA、
    cluster multicast 和 staged pipeline。构造时传入 accumulator dtype、CTA tile shape 和 cluster shape；
    调用对象时会根据实际 A/B/C tensor 派生 layout、TMA atom、shared-memory layout 和 launch grid。

    注意：
        - fp16 输入要求 A/B dtype 一致。
        - fp8、int8、uint8 输入只支持 k-major layout。
        - accumulator 对浮点输入通常是 Float32/Float16，对 int8/uint8 输入是 Int32。
        - CTA tile 和 cluster shape 必须满足 Hopper WGMMA/TMA 的约束。
    """

    # 函数 HopperWgmmaGemmKernel.__init__：初始化普通 GEMM 的静态配置，包括 accumulator dtype、CTA tile、cluster
    # shape、warp group 数、线程数和共享内存容量。 参数：self, acc_dtype, tile_shape_mn, cluster_shape_mn；返回：未显式标注。
    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric], # 比如cutlass.Float16.width
        tile_shape_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
    ):
        """
        Initializes the configuration for a Hopper dense GEMM kernel.

        This configuration includes data types for operands, tile shape, cluster configuration,
        and thread layout.

        :param acc_dtype: Data type for accumulation during computation
        :type acc_dtype: type[cutlass.Numeric]
        :param tile_shape_mn: Shape of the CTA tile (M,N)
        :type tile_shape_mn: Tuple[int, int]
        :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
        :type cluster_shape_mn: Tuple[int, int]

        中文说明：
        初始化 Hopper dense GEMM kernel 的静态配置，包括 accumulator dtype、CTA tile shape、cluster shape、
        warp-group 组织、CTA 线程数、shared-memory 容量以及后续会填充的 TMA/pipeline/layout 属性。
        """

        self.acc_dtype = acc_dtype

        self.cluster_shape_mn = cluster_shape_mn # 
        self.mma_inst_shape_mn = None
        # K dimension is deferred in _setup_attributes
        # K 维 tile 大小会在 _setup_attributes 中根据 WGMMA 形状再确定。
        self.tile_shape_mnk = (*tile_shape_mn, 1)
        # 注意： 看情况选择两个warp group
        # For large tile size, using two warp groups is preferred because using only one warp
        # 对较大的 tile，优先使用两个 warp group；只用一个 warp group 时寄存器压力更容易导致 spill。
        # group may result in register spill
        # 上一行说明的是大 tile 下单 warp group 可能带来的寄存器溢出风险。
        self.atom_layout_mnk = (
            (2, 1, 1)
            if self.tile_shape_mnk[0] > 64 and self.tile_shape_mnk[1] > 128
            else (1, 1, 1)
        )
        """
        如何理解这个atom_layout_mnk:
        
            MMA atom (WGMMA atom):  一个基础 WGMMA 操作的描述
            atom_layout: 这些 atom 在 CTA tile 里如何排布 （）
            tile shape:   最终这个 tiled_mma 覆盖多大的 M/N/K tile
            
            
        """
        
        
        self.num_mcast_ctas_a = None # TMA multicast 时，同一份 A/B tile 要广播给 cluster 内多少个 CTA。
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False
        self.tiled_mma = None

        self.occupancy = 1 # 疑问： 如何改 occupancy ： Target number of CTAs per SM (occupancy).
        self.mma_warp_groups = math.prod(self.atom_layout_mnk) # 参与wgmma的warp group的数量
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = self.mma_warp_groups * self.num_threads_per_warp_group
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90") # shared mem的大小

        self.ab_stage = None # 
        """
        ab_stage  = mainloop 里 A/B TMA load -> WGMMA consume 的 pipeline 深度
        epi_stage = epilogue 里 register -> SMEM -> GMEM store 的 pipeline 深度
        """        
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None # 可能需要从smem_capacity开始
        self.buffer_align_bytes = 1024 # alignas(1024)

    # 函数 HopperWgmmaGemmKernel._setup_attributes：根据实际输入 tensor 的 dtype/layout 派生 tiled_mma、K
    # tile、multicast、epilogue tile、pipeline stage 和 SMEM layout。 参数：self；返回：未显式标注。
    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout

        中文说明：
        根据输入 tensor 的 dtype/layout 和 kernel 配置派生运行所需属性：创建 tiled MMA，确定 K tile，
        计算 cluster layout 和 A/B multicast 数量，选择 epilogue tile，估算 A/B 与 epilogue 的 pipeline stage，
        并生成 A/B/C 在 shared memory 中的 staged layout。
        """

        # check the cta tile shape
        # 检查 CTA tile shape 是否落在该示例支持的范围内。
        if self.tile_shape_mnk[0] not in [64, 128]:
            # 遇到不支持的 dtype/layout/alignment 或运行环境时主动报错，避免继续生成非法 kernel。
            raise ValueError("CTA tile shape M must be 64/128")
        if self.tile_shape_mnk[1] not in [64, 128, 256]:
            # 遇到不支持的 dtype/layout/alignment 或运行环境时主动报错，避免继续生成非法 kernel。
            raise ValueError("CTA tile shape N must be 64/128/256")

        # 构造 Hopper WGMMA 的 tiled_mma 描述，把 A/B layout、累加类型、warp-group 组织和 MMA tile 绑定起来。
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]), #疑问： 这个地方self.tile_shape_mnk[0]?
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2]) # 取哪一个维度， mnk, 则m是mode0, k是mode2
        
        mma_inst_tile_k = 4 # inst: instruction
        self.tile_shape_mnk = ( # BM, BN, BK
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1)) # make layout 比较关键
        self.num_mcast_ctas_a = self.cluster_shape_mn[1] # 疑问：不是应该a * b吗，为什么self.cluster_shape_mn[1]对应ctas_a
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        is_cooperative = self.atom_layout_mnk == (2, 1, 1) # cooperative 在90 系列下的意思就是，>1个warp group 用于处理计算部分
        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
        )

        # Compute stage before compute smem layout
        # 先计算 pipeline stage 数，再根据 stage 维生成 shared-memory layout。
        # 下面根据 tile 大小、dtype 位宽和共享内存容量计算 pipeline stage 数；A/B stage 决定 mainloop 预取深度，epilogue stage
        # 决定写回缓冲数量。
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        # 下面一次性生成 A/B/epilogue 的 staged shared-memory layout；返回值按 A、B、C 写回顺序解包到实例属性，后续
        # TMA/WGMMA/epilogue 都会复用这些 layout。
        # 疑问： 什么情况下epilogue需要smem来存一些东西。
        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged, # A/B 的 global -> shared -> WGMMA
            self.epi_smem_layout_staged, # accumulator -> shared -> global C
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

    # 函数 HopperWgmmaGemmKernel.__call__：CuTe JIT launch 包装：读取 tensor 元信息，创建 TMA atom/tensor，定义
    # shared storage，并发起 device kernel。 参数：self, a, b, c, stream；返回：未显式标注。
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes, 静态的属性
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream

        中文说明：
        执行 GEMM 的 host/JIT 入口：先记录 A/B/C 的 dtype 和 layout，再完成合法性检查，创建 TMA load/store
        atom 与 tensor 视图，计算 launch grid，定义 shared storage，最后同步 launch device kernel。
        """

        # setup static attributes before smem/grid/tma computation
        # 在计算 SMEM/grid/TMA 之前，先记录输入 tensor 决定的静态属性。
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a) # 看是row major还是column major
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr( # a_dtype.width: bytes数
            # 编译器常量                            
            self.a_dtype.width == 16 and self.a_dtype != self.b_dtype
        ):
            # 遇到不支持的 dtype/layout/alignment 或运行环境时主动报错，避免继续生成非法 kernel。
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype.width != self.b_dtype.width):
            # 遇到不支持的 dtype/layout/alignment 或运行环境时主动报错，避免继续生成非法 kernel。
            raise TypeError(
                f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}"
            )
        if cutlass.const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            # 遇到不支持的 dtype/layout/alignment 或运行环境时主动报错，避免继续生成非法 kernel。
            raise TypeError("a_dtype should be float16 or float8")

        self._setup_attributes()

        # 创建输入矩阵的 TMA load atom 和 TMA tensor 视图；后续 kernel 用它从 global memory 异步搬到 shared memory。
        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[1],
        )

        # 创建输入矩阵的 TMA load atom 和 TMA tensor 视图；后续 kernel 用它从 global memory 异步搬到 shared memory。
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[0],
        )

        # 创建输出矩阵的 TMA store atom 和 tensor 视图；epilogue 会用它把 shared memory 中的结果写回 global memory。
        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        # 计算 kernel launch grid；persistent 版本还会同时生成 tile scheduler 参数。
        grid = self._compute_grid(c, self.tile_shape_mnk, self.cluster_shape_mn)

        # 类 HopperWgmmaGemmKernel.__call__.SharedStorage：描述普通 GEMM kernel 使用的 shared memory
        # 字段：pipeline barrier 数组以及 A/B 共享内存 buffer。
        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ] # barrier, int64
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ], # 需要用cosize，定义物理存储大小
                self.buffer_align_bytes,
            ]
            """
            cute.struct.MemRange[T, N] 就是在 struct 里声明“一段 T 类型、长度 N 的连续内存”，
            常用于 shared memory buffer；用 .data_ptr() 拿指针，用 .get_tensor(layout) 把它
            解释成 CuTe tensor。
            """
            
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        # 同步 launch kernel；这里会等 kernel launch 相关操作完成。
        # 下面是一个多行函数调用；用一段注释解释整个调用，参数行保持干净以便对照源码。
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            self.cta_layout_mnk, # cluster level
            self.a_smem_layout_staged, # 
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )
        return

    #  GPU device kernel
    #  GPU device kernel 主体
    # 函数 HopperWgmmaGemmKernel.kernel：GPU device kernel：定位输出 tile，初始化 TMA pipeline，加载 A/B，执行 WGMMA
    # mainloop，再通过 epilogue 写回 C。 参数：self, tma_atom_a, mA_mkl, tma_atom_b, mB_nkl, tma_atom_c,
    # mC_mnl, tiled_mma, cta_layout_mnk, a_smem_layout_staged, b_smem_layout_staged,
    # epi_smem_layout_staged；返回：未显式标注。
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor, # tma_tensor: GMEM和tma单元之间的坐标映射
        tma_atom_b: cute.CopyAtom, 
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_c: TMA copy atom for C tensor
        :type tma_atom_c: cute.CopyAtom
        :param mC_mnl: Output tensor C
        :type mC_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cta_layout_mnk: CTA layout
        :type cta_layout_mnk: cute.Layout
        :param a_smem_layout_staged: Shared memory layout for A
        :type a_smem_layout_staged: cute.ComposedLayout
        :param b_smem_layout_staged: Shared memory layout for B
        :type b_smem_layout_staged: cute.ComposedLayout
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout

        中文说明：
        GPU device kernel 的主体：每个 CTA 定位自己的输出 tile，预取 TMA descriptor，初始化 A/B load pipeline
        和 C store pipeline；mainloop 中用 TMA 把 A/B 搬到 SMEM，再用 WGMMA 累加到寄存器；epilogue 阶段
        把 accumulator 写入 SMEM，并通过 TMA store 写回 GMEM。
        """

        warp_idx = cute.arch.warp_idx()
        # 补充： tidx, tidy, tidz = cute.arch.thread_idx()
        # lane_idx = cute.arch.lane_idx()
        
        # 如果要thread_idx.
        # tidx, tidy, tidz = cute.arch.thread_idx()
        # bdimx, bdimy, _ = cute.arch.block_dim()
        # linear_tid = tidx + tidy * bdimx + tidz * bdimx * bdimy
        warp_idx = cute.arch.make_warp_uniform(warp_idx) # 显式告诉编译器，一个warp内warp_idx应该是一样的，避免把它当成 lane-divergent 分支处理。
        # 最好就和一般的warp_idx（）一起用

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        #  预取 TMA descriptor
        # /////////////////////////////////////////////////////////////////////////////
        # 按 warp 编号分配轻量控制工作，例如预取 TMA descriptor、发起 TMA copy 或执行 epilogue store。
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            # 疑问：为什么只拿ab，不拿c？？ 因为
            """
            dense GEMM 的 C descriptor 是“晚点用的固定 store descriptor”；grouped GEMM 的 C descriptor
            是“persistent/group 切换流程里会初始化、更新、反复 store 的动态 tensormap descriptor”，所以提
            前 prefetch C 更有收益，也更稳妥。
            """
            

        # ///////////////////////////////////////////////////////////////////////////////
        #  Get cta/warp/thread idx
        #  获取 CTA、warp 和 thread 的索引
        # ///////////////////////////////////////////////////////////////////////////////
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        # 疑问：如果cluster是(1,1) 此时cidx的swizzle还有用吗？
        # 解答： 有用，当cluster(1,1)时， cid~= bid
        cidx, cidy, _ = cute.arch.cluster_idx()
        cdimx, cdimy, _ = cute.arch.cluster_dim()
        cluster_id = cidx + cdimx * cidy

        # CTA Swizzle to promote L2 data reuse
        # 对 CTA 坐标做 swizzle，提高 L2 数据复用。
        # 注意，这里做的是cluster级别的swizzle，cluster(1,1)时，就是block级别
        # cid swizzle 的依据就是 GPU 对线性 block/cluster id 的发射顺序具有近似时间局部性；
        # swizzle 把这种“编号局部性”转换成 A/B 数据访问局部性。
        group_size_m = 8
        s_shape = (
            (group_size_m, cdimx // group_size_m), # 相当于把第一维的layout加了一层， ((m_in_group, m_group), n)
            cdimy,
        )
        s_stride = ((1, cdimy * group_size_m), group_size_m) # 因为复用目标是a、b tile，因此需要m 0-7后，重复n，即此时，n是第二维
        s_layout = cute.make_layout(s_shape, stride=s_stride)
        num_reg_cids = cute.size(s_shape)
        cid_m, cid_n = s_layout.get_flat_coord(cluster_id % num_reg_cids)
        # 一维坐标翻译成s_layout多维坐标： get_flat_coord
        # 多维转一维： linear = s_layout(coord)
        # layout(coord)          : 多维坐标 -> 线性 offset
        # layout.get_flat_coord(i): 线性 offset -> 多维坐标
        # 疑问：如果出现cluster_size // num_reg_cids >0 怎么办？
        
        
        

        # Deal with the tail part
        # 处理 M/N 维尾块，避免越界访问。
        if cluster_id >= num_reg_cids:
            tail_size_m = cdimx % group_size_m
            tail_layout = cute.make_layout(
                (tail_size_m, cdimy), stride=(1, tail_size_m)
            )
            tail_cid = cluster_id - num_reg_cids
            tail_cid_m, tail_cid_n = tail_layout.get_flat_coord(tail_cid)
            cid_m = cute.size(s_shape, mode=[0]) + tail_cid_m
            cid_n = tail_cid_n

        # Get the pid from cluster id
        # 根据 cluster id 计算当前 CTA 对应的 tile id。
        bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * self.cluster_shape_mn[0] + bidx_in_cluster[0]
        pid_n = cid_n * self.cluster_shape_mn[1] + bidx_in_cluster[1]

        tile_coord_mnkl = (pid_m, pid_n, None, bidz)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get mcast mask
        # 生成 TMA multicast mask，决定数据广播给 cluster 内哪些 CTA。
        # ///////////////////////////////////////////////////////////////////////////////
        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )

        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init AB full/empty + ACC full mbar (pipeline)
        #  分配并初始化 A/B full/empty barrier，以及 accumulator/epilogue 相关 barrier。
        # /////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # mbar arrays
        # mbar 数组保存各个 pipeline stage 的 barrier。
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # Threads/warps participating in this pipeline
        # 定义参与该 pipeline 的线程数/warp 数。
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        # Each warp will constribute to the arrive count with the number of mcast size
        # 每个 warp 对 arrive count 的贡献会乘上 multicast 的 CTA 数量。
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        num_warps = self.threads_per_cta // 32
        consumer_arrive_cnt = mcast_size * num_warps
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        # 创建 mainloop 的 TMA async pipeline，用 full/empty barrier 管理多 stage A/B shared-memory buffer。
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        #  Cluster arrive after barrier init
        #  barrier 初始化后，cluster 内 CTA 做一次 arrive 同步。
        # cluster 内 CTA 到达 pipeline 初始化同步点，确保 barrier 初始化过程可见。
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensor A/B
        #  根据 shared storage 和 layout 构造 A/B 的 SMEM tensor。
        # ///////////////////////////////////////////////////////////////////////////////
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC_ptr = cute.recast_ptr(
            sA.iterator, epi_smem_layout_staged.inner, dtype=self.c_dtype
        )
        sC = cute.make_tensor(sC_ptr, epi_smem_layout_staged.outer)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Local_tile partition global tensors
        #  从 global tensor 中切出当前 CTA/cluster 负责的 tile。
        # ///////////////////////////////////////////////////////////////////////////////
        # (bM, bK, RestK)
        # 从全局 tensor 中切出当前 tile/所有 tile 的局部视图，避免手写 M/N/K/L 索引计算。
        gA_mkl = cute.local_tile(
            mA_mkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, None, 1)
        )
        # (bN, bK, RestK)
        # 从全局 tensor 中切出当前 tile/所有 tile 的局部视图，避免手写 M/N/K/L 索引计算。
        gB_nkl = cute.local_tile(
            mB_nkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(None, 1, 1)
        )
        # (bM, bN)
        # 从全局 tensor 中切出当前 tile/所有 tile 的局部视图，避免手写 M/N/K/L 索引计算。
        gC_mnl = cute.local_tile(
            mC_mnl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, 1, None)
        )

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition global tensor for TiledMMA_A/B/C
        #  按 TiledMMA 视角对 global tensor 做分区。
        # //////////////////////////////////////////////////////////////////////////////
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

        tCgC = thr_mma.partition_C(gC_mnl)

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition shared tensor for TMA load A/B
        #  按 TMA load 需求对 shared tensor 做分区。
        # //////////////////////////////////////////////////////////////////////////////
        #  TMA load A partition_S/D
        #  为 A 的 TMA load 创建源端 S 和目的端 D 分区。
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        sA_for_tma_partition = cute.group_modes(sA, 0, 2)
        gA_for_tma_partition = cute.group_modes(gA_mkl, 0, 2)
        # 把 global/shared tensor 按 TMA atom 和 CTA/cluster 坐标分区，得到 copy 指令需要的源和目的视图。
        tAsA, tAgA_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            sA_for_tma_partition,
            gA_for_tma_partition,
        )

        # TMA load B partition_S/D
        # 为 B 的 TMA load 创建源端 S 和目的端 D 分区。
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        sB_for_tma_partition = cute.group_modes(sB, 0, 2)
        gB_for_tma_partition = cute.group_modes(gB_nkl, 0, 2)
        # 把 global/shared tensor 按 TMA atom 和 CTA/cluster 坐标分区，得到 copy 指令需要的源和目的视图。
        tBsB, tBgB_nkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            sB_for_tma_partition,
            gB_for_tma_partition,
        )

        # //////////////////////////////////////////////////////////////////////////////
        #  Make fragments
        #  创建寄存器 fragment，包括 accumulator 和临时寄存器视图。
        # //////////////////////////////////////////////////////////////////////////////
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        acc_shape = tCgC.shape
        # 创建寄存器 tensor，通常用于 accumulator、临时 accumulator 或 epilogue 类型转换缓冲。
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Cluster wait
        #  等待 cluster 级同步完成。
        # ///////////////////////////////////////////////////////////////////////////////
        # cluster wait for barrier init
        # 等待 barrier 初始化在 cluster 内可见。
        # 等待 pipeline 初始化完成，避免在 barrier 未准备好时开始 producer/consumer 操作。
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch
        #  mainloop 前的预取阶段。
        # /////////////////////////////////////////////////////////////////////////////
        k_tile_cnt = cute.size(gA_mkl, mode=[2])
        prefetch_k_tile_cnt = cutlass.max(cutlass.min(self.ab_stage, k_tile_cnt), 0)

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        # 按 warp 编号分配轻量控制工作，例如预取 TMA descriptor、发起 TMA copy 或执行 epilogue store。
        if warp_idx == 0:
            # /////////////////////////////////////////////////////////////////////////////
            # Prefetch TMA load
            # 预取首批 TMA load。
            # /////////////////////////////////////////////////////////////////////////////
            for prefetch_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                # /////////////////////////////////////////////////////////////////////////////
                #  Wait for A/B buffers to be empty before loading into them
                #  写入 A/B buffer 前，先等待对应 pipeline stage 为空。
                #  Also sets the transaction barrier for the A/B buffers
                #  同时设置 A/B buffer 对应的 transaction barrier。
                # /////////////////////////////////////////////////////////////////////////////
                # producer 等待目标 pipeline stage 变空，准备把新的 A/B tile 通过 TMA 搬入 shared memory。
                mainloop_pipeline.producer_acquire(mainloop_producer_state)
                # /////////////////////////////////////////////////////////////////////////////
                #  Slice to global/shared memref to current k_tile
                #  切出当前 k_tile 对应的 global/shared memref。
                # /////////////////////////////////////////////////////////////////////////////
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                # /////////////////////////////////////////////////////////////////////////////
                #  TMA load A/B
                #  发起 A/B 的 TMA load。
                # /////////////////////////////////////////////////////////////////////////////
                # 执行 CuTe copy；根据上下文可能是 TMA load、TMA store 或寄存器到 shared memory 的 copy。
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=a_mcast_mask,
                )
                # 执行 CuTe copy；根据上下文可能是 TMA load、TMA store 或寄存器到 shared memory 的 copy。
                cute.copy(
                    tma_atom_b,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=b_mcast_mask,
                )
                # Mainloop pipeline's producer commit is a NOP
                # mainloop pipeline 的 producer commit 在这里是空操作，但保留统一的 pipeline 语义。
                # producer 提交当前 pipeline stage；TMA async pipeline 中它主要推进状态语义。
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        # /////////////////////////////////////////////////////////////////////////////
        #  Prologue MMAs
        #  prologue 阶段先发起一批 MMA，填充 WGMMA pipeline。
        # /////////////////////////////////////////////////////////////////////////////
        k_pipe_mmas = 1

        mainloop_consumer_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        mainloop_consumer_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        peek_ab_full_status = cutlass.Boolean(1)
        if mainloop_consumer_read_state.count < k_tile_cnt:
            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                mainloop_consumer_read_state
            )

        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        num_k_blocks = cute.size(tCrA, mode=[2])
        # 沿 K 维 tile 迭代 mainloop；每轮消费一个 K tile 的 A/B 数据并贡献一部分矩阵乘加。
        for k_tile in cutlass.range_constexpr(k_pipe_mmas):
            # Wait for A/B buffer to be ready
            # 等待 A/B buffer 中的数据准备好。
            # consumer 等待当前 pipeline stage 的 TMA load 完成，确保 WGMMA 读取有效的 shared-memory 数据。
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_ab_full_status
            )

            cute.nvgpu.warpgroup.fence()
            # 遍历当前 K tile 内的 WGMMA K-block；每个 block 发起一次 CuTe GEMM/WGMMA。
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (
                    None,
                    None,
                    k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                tCrA_1phase = tCrA[k_block_coord]
                tCrB_1phase = tCrB[k_block_coord]

                # 发起一次 CuTe GEMM/WGMMA，把当前 A/B fragment 累加到 accumulator。
                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA_1phase,
                    tCrB_1phase,
                    accumulators,
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            # 提交当前 WGMMA group，让异步矩阵乘加进入执行队列。
            cute.nvgpu.warpgroup.commit_group()
            mainloop_consumer_read_state.advance()
            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )

        # /////////////////////////////////////////////////////////////////////////////
        #  MAINLOOP
        #  主循环
        # /////////////////////////////////////////////////////////////////////////////
        # 沿 K 维 tile 迭代 mainloop；每轮消费一个 K tile 的 A/B 数据并贡献一部分矩阵乘加。
        for k_tile in cutlass.range(k_pipe_mmas, k_tile_cnt, 1, unroll=1):
            # /////////////////////////////////////////////////////////////////////////////
            #  Wait for TMA copies to complete
            #  等待当前 stage 的 TMA copy 完成。
            # /////////////////////////////////////////////////////////////////////////////
            # consumer 等待当前 pipeline stage 的 TMA load 完成，确保 WGMMA 读取有效的 shared-memory 数据。
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_ab_full_status
            )
            # /////////////////////////////////////////////////////////////////////////////
            #  WGMMA
            #  发起当前 K tile 内的 WGMMA。
            # /////////////////////////////////////////////////////////////////////////////
            cute.nvgpu.warpgroup.fence()
            # 遍历当前 K tile 内的 WGMMA K-block；每个 block 发起一次 CuTe GEMM/WGMMA。
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (
                    None,
                    None,
                    k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                tCrA_1phase = tCrA[k_block_coord]
                tCrB_1phase = tCrB[k_block_coord]

                # 发起一次 CuTe GEMM/WGMMA，把当前 A/B fragment 累加到 accumulator。
                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA_1phase,
                    tCrB_1phase,
                    accumulators,
                )

            # 提交当前 WGMMA group，让异步矩阵乘加进入执行队列。
            cute.nvgpu.warpgroup.commit_group()
            # Wait on the wgmma barrier for previous k_pipe_mmas wgmmas to complete
            # 等待前面提交的一批 WGMMA 完成。
            # 等待 WGMMA group 完成；在读取 accumulator 或释放 buffer 前必须保证写入结束。
            cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)

            # consumer 释放已经用完的 pipeline stage，让 producer 后续可以复用该 shared-memory buffer。
            mainloop_pipeline.consumer_release(mainloop_consumer_release_state)

            mainloop_consumer_read_state.advance()
            # consumer 释放已经用完的 pipeline stage，让 producer 后续可以复用该 shared-memory buffer。
            mainloop_consumer_release_state.advance()

            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )
            # /////////////////////////////////////////////////////////////////////////////
            #  TMA load
            # /////////////////////////////////////////////////////////////////////////////
            # 按 warp 编号分配轻量控制工作，例如预取 TMA descriptor、发起 TMA copy 或执行 epilogue store。
            if warp_idx == 0 and mainloop_producer_state.count < k_tile_cnt:
                # /////////////////////////////////////////////////////////////////////////////
                #  Wait for A/B buffers to be empty before loading into them
                #  写入 A/B buffer 前，先等待对应 pipeline stage 为空。
                #  Also sets the transaction barrier for the A/B buffers
                #  同时设置 A/B buffer 对应的 transaction barrier。
                # /////////////////////////////////////////////////////////////////////////////
                # producer 等待目标 pipeline stage 变空，准备把新的 A/B tile 通过 TMA 搬入 shared memory。
                mainloop_pipeline.producer_acquire(mainloop_producer_state)

                # /////////////////////////////////////////////////////////////////////////////
                #  Slice to global/shared memref to current k_tile
                #  切出当前 k_tile 对应的 global/shared memref。
                # /////////////////////////////////////////////////////////////////////////////
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                # /////////////////////////////////////////////////////////////////////////////
                #  TMA load A/B
                #  发起 A/B 的 TMA load。
                # /////////////////////////////////////////////////////////////////////////////
                # 执行 CuTe copy；根据上下文可能是 TMA load、TMA store 或寄存器到 shared memory 的 copy。
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=a_mcast_mask,
                )
                # 执行 CuTe copy；根据上下文可能是 TMA load、TMA store 或寄存器到 shared memory 的 copy。
                cute.copy(
                    tma_atom_b,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=b_mcast_mask,
                )
                # Mainloop pipeline's producer commit is a NOP
                # mainloop pipeline 的 producer commit 在这里是空操作，但保留统一的 pipeline 语义。
                # producer 提交当前 pipeline stage；TMA async pipeline 中它主要推进状态语义。
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        # /////////////////////////////////////////////////////////////////////////////
        #  EPILOG
        #  epilogue 写回阶段
        # /////////////////////////////////////////////////////////////////////////////
        # 等待 WGMMA group 完成；在读取 accumulator 或释放 buffer 前必须保证写入结束。
        cute.nvgpu.warpgroup.wait_group(0)

        if cute.size(self.cluster_shape_mn) > 1:
            # Wait for all threads in the cluster to finish, avoid early release of smem
            # 等待 cluster 内所有线程完成，避免过早释放或复用 SMEM。
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        else:
            # For cluster that has a single thread block, it might have more than one warp groups.
            # 即使 cluster 只有一个 CTA，一个 CTA 内也可能有多个 warp group。
            # Wait for all warp groups in the thread block to finish, because smem for tensor A in
            # 等待 CTA 内所有 warp group 完成，因为 mainloop 中 A 使用的 SMEM 会在 epilogue 中复用。
            # the mainloop is reused in the epilogue.
            # 上一行说明的是 mainloop SMEM 与 epilogue SMEM 的复用关系。
            cute.arch.sync_threads()

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            self.c_layout,
            elem_ty_d=self.c_dtype,
            elem_ty_acc=self.acc_dtype,
        )

        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                self.c_layout.is_m_major_c(),
                4,
            ),
            self.c_dtype,
        )

        tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

        tiled_copy_r2s = cute.make_tiled_copy_S(
            copy_atom_r2s,
            tiled_copy_C_Atom,
        )

        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rAcc = tiled_copy_r2s.retile(accumulators)

        # Allocate D registers.
        # 分配 D 寄存器，用于存放转换后的输出片段。
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        # 创建寄存器 tensor，通常用于 accumulator、临时 accumulator 或 epilogue 类型转换缓冲。
        tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, self.acc_dtype)
        size_tRS_rD = cute.size(tRS_rD)

        sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
        tCgC_for_tma_partition = cute.zipped_divide(gC_mnl, self.epi_tile)

        # 把 global/shared tensor 按 TMA atom 和 CTA/cluster 坐标分区，得到 copy 指令需要的源和目的视图。
        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sepi_for_tma_partition,
            tCgC_for_tma_partition,
        )

        epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
        epi_tile_shape = tCgC_for_tma_partition.shape[1]
        epi_tile_layout = cute.make_layout(
            epi_tile_shape, stride=(epi_tile_shape[1], 1)
        )

        # Initialize tma store c_pipeline
        # 初始化 C 的 TMA store pipeline。
        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.threads_per_cta
        )
        # 创建 epilogue 的 TMA store pipeline，用来协调 shared-to-global 异步写回。
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=c_producer_group,
        )

        # 遍历 epilogue 子 tile，把 accumulator 分块转换、写入 shared memory，再通过 TMA store 写回输出。
        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            # Copy from accumulators to D registers
            # 从 accumulator 拷贝到 D 寄存器。
            # 遍历 epilogue 子 tile，把 accumulator 分块转换、写入 shared memory，再通过 TMA store 写回输出。
            for epi_v in cutlass.range_constexpr(size_tRS_rD):
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

            # Type conversion
            # 做 accumulator dtype 到输出 dtype 的类型转换。
            # 创建寄存器 tensor，通常用于 accumulator、临时 accumulator 或 epilogue 类型转换缓冲。
            tRS_rD_out = cute.make_rmem_tensor_like(tRS_rD_layout, self.c_dtype)
            acc_vec = tRS_rD.load()
            tRS_rD_out.store(acc_vec.to(self.c_dtype))

            # Copy from D registers to shared memory
            # 将 D 寄存器中的结果写入 shared memory。
            epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
            # 执行 CuTe copy；根据上下文可能是 TMA load、TMA store 或寄存器到 shared memory 的 copy。
            cute.copy(
                tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)]
            )

            # 下面是一个多行函数调用；用一段注释解释整个调用，参数行保持干净以便对照源码。
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            # barrier for sync
            # 用 barrier 同步，保证 SMEM 中的数据可被 TMA store 安全读取。
            pipeline.sync(barrier_id=1)

            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            # Copy from shared memory to global memory
            # 使用 TMA store 将结果从 shared memory 写回 global memory。
            # 按 warp 编号分配轻量控制工作，例如预取 TMA descriptor、发起 TMA copy 或执行 epilogue store。
            if warp_idx == 0:
                # 执行 CuTe copy；根据上下文可能是 TMA load、TMA store 或寄存器到 shared memory 的 copy。
                cute.copy(
                    tma_atom_c,
                    bSG_sD[(None, epi_buffer)],
                    bSG_gD[(None, gmem_coord)],
                )
                # producer 提交当前 pipeline stage；TMA async pipeline 中它主要推进状态语义。
                c_pipeline.producer_commit()
                # producer 等待目标 pipeline stage 变空，准备把新的 A/B tile 通过 TMA 搬入 shared memory。
                c_pipeline.producer_acquire()

            pipeline.sync(barrier_id=1)

        # 按 warp 编号分配轻量控制工作，例如预取 TMA descriptor、发起 TMA copy 或执行 epilogue store。
        if warp_idx == 0:
            # producer 结束 pipeline，通知 consumer 不会再有新的 TMA stage。
            c_pipeline.producer_tail()

        return

    # 函数 HopperWgmmaGemmKernel._compute_stages：根据 tile shape、dtype 宽度、SMEM 容量和 occupancy 估算 A/B
    # pipeline stage 数。 参数：tile_shape_mnk, a_dtype, b_dtype, smem_capacity, occupancy；返回：tuple[int,
    # int]。
    # 通常如何估算一个gemm需要的stages数量？ 总的smem。 每个阶段tma的量，包括a/b矩阵的bm，bn，bk大小，最后的c矩阵的大小，
    # 以及中间可能能复用的smem。不要spill
    # NOTE: 需要关注epi_tile 的影响
    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: tuple[int, int]

        中文说明：
        根据 CTA tile 大小、A/B dtype 位宽、shared-memory 容量和目标 occupancy，估算 A/B mainloop 可以放下
        多少个 pipeline stage；epilogue stage 在这里固定为 4，并假设 epilogue SMEM 复用 A/B 的空间。
        """

        # 一般情况下，epi阶段可以直接使用前面a/b的smem
        epi_stage = 4
        # epi_smem will reuse smem ab.
        # epilogue 的 SMEM 会复用 A/B mainloop 的 SMEM 空间。
        epi_bytes = 0 # 疑问：为什么是0？

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None)) # bm, bk
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None)) # bn, bk
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8 #  cute.size通常来算总和， // 8的原因是，求的是bytes，不是bits
            + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024 # barrier，通常是一个stage一个barrier， 通常 1 个 mbarrier = 1 个 Int64 = 8 bytes

        ab_stage = (
            smem_capacity // occupancy - mbar_helpers_bytes - epi_bytes
        ) // ab_bytes_per_stage # 疑问： 最后结果不需要暂存？所以只需要ab size即可？ 解答： 可能只需要一个accumulator
        return ab_stage, epi_stage

    # 函数 HopperWgmmaGemmKernel._make_smem_layouts：创建 A、B 和 epilogue C 的 staged shared-memory layout。
    # 参数：tile_shape_mnk, epi_tile, a_dtype, a_layout, b_dtype, b_layout, ab_stage, c_dtype,
    # c_layout, epi_stage；返回：tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]。
    # 注意： 这里就是为了tma和计算对应的layout
    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int], # BM, BN, BK
        epi_tile: tuple[int, int], # 大概率是BM, BN
        a_dtype: type[cutlass.Numeric], # 注意： 输入tensor的dtype是cutlass.Numeric
        a_layout: utils.LayoutEnum, # 注意： layout通常是row-major或者column-major
        # 疑问： 为什么cute多数是column - major
        b_dtype: type[cutlass.Numeric],
        b_layout: utils.LayoutEnum,
        ab_stage: int,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        """Create shared memory layouts for A, B, and C tensors.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout enum for matrix A
        :type a_layout: utils.LayoutEnum
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout enum for matrix B
        :type b_layout: utils.LayoutEnum
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param c_dtype: Data type for output matrix C
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum for the output matrix C
        :type c_layout: utils.LayoutEnum
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]

        中文说明：
        为 A、B 和 epilogue C 分别创建 shared-memory layout。A/B layout 用于 TMA load 和 WGMMA 读取，
        C 的 epilogue layout 用于 accumulator 先落到 SMEM，再由 TMA store 写回 GMEM。返回的 layout 都带 stage 维。
        """
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout,
            tile_shape_mnk,
            a_dtype,
            ab_stage,
        )

        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout,
            tile_shape_mnk,
            b_dtype,
            ab_stage,
        )

        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            epi_stage,
        )

        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    # 函数 HopperWgmmaGemmKernel._compute_grid：按 CTA tile 和 cluster shape 把输出 C 分块，得到 kernel launch
    # grid。 参数：c, tile_shape_mnk, cluster_shape_mn；返回：tuple[int, int, int]。
    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
    ) -> tuple[int, int, int]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]

        :return: Grid shape for kernel launch.
        :rtype: tuple[int, int, int]

        中文说明：
        按 CTA tile shape 将输出 C 切成 tile，再按 cluster shape 对 tile grid 做分组，最终得到 kernel launch
        使用的三维 grid。第三维通常对应 batch 维 L。
        这里的 grid 基本就是“所有 CTA tile 的网格”；在这个 dense GEMM 中，一个 CTA 负责一个输出 tile BM x BN，
        但 grid 会按 cluster 尺寸向上补齐。
        """

        c_shape = (tile_shape_mnk[0], tile_shape_mnk[1])
        gc = cute.zipped_divide(c, tiler=c_shape) # 用的是zipped divide, 结果是(tile内的坐标， tile的坐标)
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        clusters = cute.ceil_div(cute.get(gc.layout, mode=[1]).shape, cluster_shape_mnl)# tile的坐标， 结果是cluster的每个维度有多少个cluster group
        grid = tuple(x * y for x, y in zip(clusters, cluster_shape_mnl)) #  #cluster_group_count * #tiles_in_clusters
        return grid

    # 函数 HopperWgmmaGemmKernel._make_tma_store_atoms_and_tensors：创建 C 的 TMA shared-to-global store
    # atom 和对应 tensor 视图。 参数：tensor_c, epi_smem_layout_staged, epi_tile；返回：tuple[cute.CopyAtom,
    # cute.Tensor]。
    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for C tensor storage.

        :param tensor_c: Output tensor C
        :type tensor_c: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]

        中文说明：
        创建 C 的 TMA store atom 和对应 tensor 视图。这里使用 shared-to-global 的 bulk tensor tile store，
        负责把 epilogue 阶段暂存在 SMEM 中的 C tile 写回 global memory。
        """
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        # 下面是一次多返回值解包：把右侧计算结果拆成 (tma_atom_c, tma_tensor_c)，多行参数保持原代码结构，不逐行解释。
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )

        return tma_atom_c, tma_tensor_c

    # 函数 HopperWgmmaGemmKernel._make_tma_atoms_and_tensors：创建 A/B 的 TMA global-to-shared load
    # atom；cluster 维度大于 1 时启用 multicast。 参数：tensor, smem_layout_staged, smem_tile,
    # mcast_dim；返回：tuple[cute.CopyAtom, cute.Tensor]。
    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout_staged: Shared memory layout for the tensor
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]

        中文说明：
        创建 A 或 B 的 TMA load atom 和 tensor 视图。普通情况使用 global-to-shared TMA copy；当 mcast_dim
        大于 1 时使用 TMA multicast，让一个 CTA 发起的数据搬运可以被同一个 cluster 内多个 CTA 共享。
        """
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        # 下面是一次多返回值解包：把右侧计算结果拆成 (tma_atom, tma_tensor)，多行参数保持原代码结构，不逐行解释。
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        # tma_atom:  # 一次 TMA 拷贝操作的 基础描述单元。包括metadata，以及一些具体的tensor
        # tma_tensor: GMEM和tma单元之间的坐标映射
        return tma_atom, tma_tensor

    # 函数 HopperWgmmaGemmKernel.is_valid_dtypes：检查输入、累加和输出 dtype 组合，以及 8-bit 输入的 layout 约束。
    # 参数：a_dtype, b_dtype, acc_dtype, c_dtype, a_major, b_major；返回：bool。
    @staticmethod
    def is_valid_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
    ) -> bool:
        """
        Check if the dtypes are valid

        :param a_dtype: The data type of tensor A
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of tensor B
        :type b_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: major mode of tensor A
        :type a_major: str
        :param b_major: major mode of tensor B
        :type b_major: str

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool

        中文说明：
        检查 A/B/C 和 accumulator 的 dtype 组合是否合法。重点约束包括：A/B 必须是支持的输入类型；
        fp16 的 A/B dtype 必须相同；A/B 位宽必须相同；8-bit 输入只允许 k-major layout；整数输入使用 Int32 累加。
        """
        is_valid = True

        valid_ab_dtypes = {
            cutlass.Float16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
            cutlass.Uint8,
            cutlass.Int8,
        }
        if a_dtype not in valid_ab_dtypes:
            is_valid = False
        if b_dtype not in valid_ab_dtypes:
            is_valid = False

        # make sure a_dtype == b_dtype for Float16
        # Float16 路径要求 A/B dtype 完全相同。
        if a_dtype.width == 16 and a_dtype != b_dtype:
            is_valid = False
        if a_dtype.width != b_dtype.width:
            is_valid = False
        if not a_dtype.is_same_kind(b_dtype):
            is_valid = False

        # for 8-bit types, this implementation only supports k-major layout
        # 对 8-bit 类型，该实现只支持 k-major layout。
        if (a_dtype.width == 8 and a_major != "k") or (
            b_dtype.width == 8 and b_major != "k"
        ):
            is_valid = False

        # Define compatibility mapping between accumulator type and AB type
        # 定义 accumulator dtype 与 A/B dtype 的兼容关系。
        acc_ab_compatibility = {
            cutlass.Float32: {
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Float16: {
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Int32: {cutlass.Uint8, cutlass.Int8},
        }
        # Check compatibility between accumulator type and A type
        # 检查 accumulator dtype 是否兼容 A dtype。
        if a_dtype not in acc_ab_compatibility[acc_dtype]:
            is_valid = False

        # Define compatibility mapping between accumulator type and C type
        # 定义 accumulator dtype 与 C dtype 的兼容关系。
        acc_c_compatibility = {
            cutlass.Float32: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Float16: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Int32: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.Int32,
                cutlass.Int8,
                cutlass.Uint8,
            },
        }
        # Check compatibility between accumulator type and C type
        # 检查 accumulator dtype 是否兼容 C dtype。
        if c_dtype not in acc_c_compatibility[acc_dtype]:
            is_valid = False

        return is_valid

    # 函数 HopperWgmmaGemmKernel.is_valid_tensor_alignment：检查 A/B/C 连续维是否满足 TMA 需要的 16 字节对齐。 参数：m, n,
    # k, l, ab_dtype, c_dtype, a_major, b_major, c_major；返回：bool。
    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool

        中文说明：
        检查 A/B/C 的连续维是否满足 TMA 需要的 16B 对齐。这里根据 dtype 位宽算出 16 字节对应多少个元素，
        再要求每个 tensor 的 major/contiguous 维长度是该元素数的整数倍。
        """
        is_valid = True

        # 函数
        # HopperWgmmaGemmKernel.is_valid_tensor_alignment.check_contigous_16B_alignment：这一段定义一个局部作用域，用来组织当前
        # GEMM 示例的配置、数据搬运、计算或校验逻辑。 参数：dtype, is_mode0_major, tensor_shape；返回：未显式标注。
        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid


# 函数 run：完整 host 示例入口：构造 torch/CuTe tensor，编译 kernel，执行可选参考校验并 benchmark。 参数：mnkl, a_dtype, b_dtype,
# c_dtype, acc_dtype, a_major, b_major, c_major, tile_shape_mn, cluster_shape_mn, tolerance,
# warmup_iterations, iterations, skip_ref_check, use_cold_l2, **kwargs；返回：未显式标注。
def run(
    mnkl: Tuple[int, int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    tile_shape_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float,
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    use_cold_l2: bool = False,
    **kwargs,
):
    """
    Prepare A/B/C tensors, launch GPU kernel, and reference checking.

    :param mnkl: Problem size (M, N, K, L)
    :type mnkl: Tuple[int, int, int, int]
    :param a_dtype: Data type for input tensor A
    :type a_dtype: Type[cutlass.Numeric]
    :param b_dtype: Data type for input tensor B
    :type b_dtype: Type[cutlass.Numeric]
    :param c_dtype: Data type for output tensor C
    :type c_dtype: Type[cutlass.Numeric]
    :param acc_dtype: Data type for accumulation during matrix multiplication
    :type acc_dtype: Type[cutlass.Numeric]
    :param a_major/b_major/c_major: Memory layout of tensor A/B/C
    :type a_major/b_major/c_major: str
    :param tile_shape_mn: CTA tile shape (M, N)
    :type tile_shape_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster shape (M, N)
    :type cluster_shape_mn: Tuple[int, int]
    :param tolerance: Tolerance value for reference validation comparison
    :type tolerance: float
    :param warmup_iterations: Number of warmup iterations before benchmarking, defaults to 0
    :type warmup_iterations: int, optional
    :param iterations: Number of benchmark iterations to run, defaults to 1
    :type iterations: int, optional
    :param skip_ref_check: Whether to skip reference result validation, defaults to False
    :type skip_ref_check: bool, optional
    :param use_cold_l2: Whether to use circular buffer strategy to ensure cold L2 cache, defaults to False
    :type use_cold_l2: bool, optional
    :return: Execution time of the GEMM kernel in microseconds
    :rtype: float

    中文说明：
    完整 host 示例入口：解析问题规模和 layout，创建 torch tensor 并转换成 CuTe tensor，构造/编译 kernel，
    launch GPU GEMM；如果没有跳过参考校验，会用 torch 结果做正确性检查；最后按 warmup/iteration 参数做 benchmark，
    返回 kernel 执行时间，单位是微秒。
    """

    import torch
    import cutlass.torch as cutlass_torch

    print("Running Hopper Dense GEMM with:")
    print(f"mnkl: {mnkl}")
    # 下面是一个多行函数调用；用一段注释解释整个调用，参数行保持干净以便对照源码。
    print(
        f"A dtype: {a_dtype}, B dtype: {b_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}"
    )
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Tile Shape: {tile_shape_mn}, Cluster Shape: {cluster_shape_mn}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {use_cold_l2}")

    # Unpack parameters
    # 解包问题规模参数。
    m, n, k, l = mnkl

    # 运行前合法性检查：不支持的 dtype/layout/alignment 或无 GPU 环境会提前报错。
    if not HopperWgmmaGemmKernel.is_valid_dtypes(
        a_dtype, b_dtype, acc_dtype, c_dtype, a_major, b_major
    ):
        # 遇到不支持的 dtype/layout/alignment 或运行环境时主动报错，避免继续生成非法 kernel。
        raise TypeError(
            f"unsupported combination of types and majors: A {a_dtype}, B {b_dtype}, Acc {acc_dtype}, C {c_dtype}, {a_major=}, {b_major=}"
        )
    # 运行前合法性检查：不支持的 dtype/layout/alignment 或无 GPU 环境会提前报错。
    if not HopperWgmmaGemmKernel.is_valid_tensor_alignment(
        m, n, k, l, a_dtype, c_dtype, a_major, b_major, c_major
    ):
        # 遇到不支持的 dtype/layout/alignment 或运行环境时主动报错，避免继续生成非法 kernel。
        raise TypeError(
            "the contiguous dimension of A/B/C tensors is not 16 bytes aligned"
        )

    # 运行前合法性检查：不支持的 dtype/layout/alignment 或无 GPU 环境会提前报错。
    if not torch.cuda.is_available():
        # 遇到不支持的 dtype/layout/alignment 或运行环境时主动报错，避免继续生成非法 kernel。
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # Create and permute tensor A/B/C
    # 创建并按目标 layout 变换 A/B/C tensor。
    # 函数 run.create_and_permute_tensor：按目标 major layout 创建 tensor，转换成 CuTe tensor，并保留 f32 版本用于参考结果。
    # 参数：l, mode0, mode1, is_mode0_major, dtype, is_dynamic_layout；返回：未显式标注。
    def create_and_permute_tensor(
        l, mode0, mode1, is_mode0_major, dtype, is_dynamic_layout=True
    ):
        # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
        # 如果 mode0 是连续主维，先创建 (l, mode1, mode0)，再 permute 成 (mode0, mode1, l)。
        # else : (l, mode0, mode1) -> (mode0, mode1, l)
        # 否则先创建 (l, mode0, mode1)，再 permute 成 (mode0, mode1, l)。
        shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
        permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
        is_unsigned = dtype in {cutlass.Uint8}
        # Temporarily use uint8 as torch does not support fp8 type
        # torch 暂不直接支持 fp8 tensor 创建，这里临时用 uint8 承载 fp8 数据。
        torch_dtype = (
            cutlass_torch.dtype(dtype)
            if dtype not in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}
            else torch.uint8
        )

        # Create dtype torch tensor (cpu)
        # 在 CPU 上创建目标 dtype 的 torch tensor。
        torch_tensor_cpu = cutlass.torch.create_and_permute_torch_tensor(
            shape,
            torch_dtype,
            permute_order=permute_order,
            init_type=cutlass.torch.TensorInitType.RANDOM,
            init_config=cutlass.torch.RandomInitConfig(
                min_val=0 if is_unsigned else -2, max_val=4 if is_unsigned else 2
            ),
        )
        # Create dtype torch tensor (gpu)
        # 将目标 dtype tensor 搬到 GPU。
        torch_tensor = torch_tensor_cpu.cuda()

        # Create f32 torch tensor (cpu)
        # 在 CPU 上保留 f32 版本，供参考计算或类型转换使用。
        f32_torch_tensor = torch_tensor_cpu.to(dtype=torch.float32)

        # Create dtype cute tensor (gpu)
        # 从 GPU torch tensor 创建 CuTe tensor 视图。
        cute_tensor = from_dlpack(torch_tensor, assumed_align=16)
        cute_tensor.element_type = dtype
        if is_dynamic_layout:
            cute_tensor = cute_tensor.mark_layout_dynamic(
                leading_dim=(0 if is_mode0_major else 1)
            )
        cute_tensor = cutlass.torch.convert_cute_tensor(
            f32_torch_tensor,
            cute_tensor,
            dtype,
            is_dynamic_layout=is_dynamic_layout,
        )

        return f32_torch_tensor, cute_tensor, torch_tensor

    a, mA, a_torch = create_and_permute_tensor(l, m, k, a_major == "m", a_dtype)
    b, mB, b_torch = create_and_permute_tensor(l, n, k, b_major == "n", b_dtype)
    c, mC, c_torch = create_and_permute_tensor(l, m, n, c_major == "m", c_dtype)

    gemm = HopperWgmmaGemmKernel(acc_dtype, tile_shape_mn, cluster_shape_mn)

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    # compile gemm kernel
    # 编译 GEMM kernel。
    # 触发 CuTe DSL JIT 编译，把 Python kernel 描述和示例参数 specialize 成可 launch 的 CUDA kernel。
    compiled_gemm = cute.compile(gemm, mA, mB, mC, stream)

    # 根据命令行选项决定是否跳过参考校验；跳过会更快，但不会验证输出正确性。
    if not skip_ref_check:
        # execution
        # 执行编译后的 kernel。
        compiled_gemm(mA, mB, mC, stream)

        torch.cuda.synchronize()

        # Ref check
        # 参考结果校验。
        # 用 PyTorch 计算参考 GEMM 结果，用于和自定义 kernel 输出做数值校验。
        ref = (torch.einsum("mkl,nkl->mnl", a, b)).cpu()

        if c_dtype in (cutlass.Float8E4M3FN, cutlass.Float8E5M2):
            # m major: (l, n, m) -> (m, n, l)
            # m-major 输出先按 (l, n, m) 创建，再 permute 到 (m, n, l)。
            # n major: (l, m, n) -> (m, n, l)
            # n-major 输出先按 (l, m, n) 创建，再 permute 到 (m, n, l)。
            permute_order = (1, 2, 0) if c_major == "n" else (2, 1, 0)
            shape = (l, m, n) if c_major == "n" else (l, n, m)
            f8_torch_tensor = cutlass_torch.create_and_permute_torch_tensor(
                shape,
                torch.uint8,
                permute_order=permute_order,
                init_type=cutlass_torch.TensorInitType.SKIP,
            ).cuda()
            # Create dtype cute tensor (gpu)
            ref_c_tensor = from_dlpack(
                f8_torch_tensor, assumed_align=16
            ).mark_layout_dynamic(leading_dim=(1 if c_major == "n" else 0))
            ref_c_tensor.element_type = c_dtype
            ref_c_tensor = cutlass_torch.convert_cute_tensor(
                ref,
                ref_c_tensor,
                c_dtype,
                is_dynamic_layout=True,
            )
            ref_c = f8_torch_tensor.cpu()
        else:
            ref_c = ref.to(cutlass_torch.dtype(c_dtype))

        # 比较 kernel 输出和 PyTorch 参考结果，容差内则认为数值正确。
        torch.testing.assert_close(c_torch.cpu(), ref_c, atol=tolerance, rtol=1e-03)

    # 函数 run.generate_tensors：benchmark workspace 生成器，用于普通测量或 cold L2 测量。 参数：无；返回：未显式标注。
    def generate_tensors():
        _, mA_workspace, _ = create_and_permute_tensor(l, m, k, a_major == "m", a_dtype)
        _, mB_workspace, _ = create_and_permute_tensor(l, n, k, b_major == "n", b_dtype)
        _, mC_workspace, _ = create_and_permute_tensor(l, m, n, c_major == "m", c_dtype)
        return testing.JitArguments(mA_workspace, mB_workspace, mC_workspace, stream)

    workspace_count = 1
    # 根据选项启用 cold L2 benchmark 策略，通过轮换 workspace 降低缓存复用影响。
    if use_cold_l2:
        one_workspace_bytes = (
            a_torch.numel() * a_torch.element_size()
            + b_torch.numel() * b_torch.element_size()
            + c_torch.numel() * c_torch.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    # 调用 CUTLASS testing benchmark 多次运行 compiled kernel，并返回执行时间。
    exec_time = testing.benchmark(
        compiled_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    return exec_time  # Return execution time in microseconds


# 脚本入口保护：只有直接运行该文件时才解析命令行并调用 run。
if __name__ == "__main__":
    args = parse_arguments()
    # 下面是一个多行函数调用；用一段注释解释整个调用，参数行保持干净以便对照源码。
    run(
        args.mnkl,
        args.a_dtype,
        args.b_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.tile_shape_mn,
        args.cluster_shape_mn,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )
    print("PASS")
