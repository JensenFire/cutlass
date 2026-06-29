# -*- coding: utf-8 -*-
# 中文块级注释版，原始文件：dense_gemm_fp8_2xacc.py
# FP8 2xAcc Hopper dense GEMM：固定 FP8 E4M3 输入和 Float32 累加，并在 epilogue 应用 scale_a * scale_b。
# 2xAcc 使用 accum_temp + accumulators 两级累加，按 mma_promotion_interval 周期性提升临时累加结果。
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
from typing import Optional, Tuple, Type
import math
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils

"""
A high-performance FP8 GEMM (D = scale_a * scale_b * A * B) example for the NVIDIA Hopper
architecture using CuTe DSL, featuring the 2xAcc (double accumulation) technique for improved
FP8 numerical accuracy.

This is a CuTeDSL port of CUTLASS Example 54:
  examples/54_hopper_fp8_warp_specialized_gemm/54_hopper_fp8_warp_specialized_gemm.cu

The 2xAcc technique addresses FP8 precision loss by maintaining two accumulators:
  - accum_temp: A temporary accumulator that WGMMA writes into directly
  - accum: The main accumulator that collects promoted partial results

Every `mma_promotion_interval` MMA instructions, accum_temp is promoted (element-wise added)
into accum, then reset to zero for the next batch of MMAs. This periodic promotion prevents
precision degradation from accumulating too many low-precision FP8 products.

The C++ reference for the 2xAcc algorithm is in:
  include/cutlass/gemm/collective/fp8_accumulation.hpp
  include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8.hpp

- Matrix A is MxKxL (FP8 E4M3, k-major only)
- Matrix B is NxKxL (FP8 E4M3, k-major only)
- Matrix D is MxNxL (configurable output dtype)

This GEMM kernel supports the following features:
    - FP8 (E4M3FN) inputs with Float32 accumulation
    - 2xAcc (double accumulation) for improved FP8 numerical accuracy
    - Scalar scale_a and scale_b factors applied in the epilogue
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Hopper's WGMMA for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with MMA between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and MMA

To run this example:

.. code-block:: bash

    python examples/python/CuTeDSL/hopper/dense_gemm_fp8_2xacc.py             \
      --mnkl 2048,2048,2048,1 --tile_shape_mn 128,128                          \
      --cluster_shape_mn 1,2 --mma_promotion_interval 4                        \
      --c_dtype Float16 --scale_a 1.0 --scale_b 1.0

Constraints:
* Input data types: FP8 E4M3FN only, k-major layout
* Accumulation dtype: Float32
* Output dtype: Float16, Float32, or Float8E4M3FN
* CTA tile shape M must be 64/128
* CTA tile shape N must be 64/128/256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 4
* The contiguous dimension of tensors must be at least 16 bytes aligned (16 elements for FP8)
* mma_promotion_interval must be a multiple of num_k_blocks per k_tile (typically 4)

中文说明：
这是 Hopper 上的 FP8 2xAcc GEMM 示例，计算 D = scale_a * scale_b * A * B。A/B 固定为 FP8 E4M3FN 且只支持 k-major，accumulator 使用 Float32。
2xAcc 的核心是维护两个累加器：accum_temp 接收 WGMMA 直接写入的临时部分和，accum 保存周期性提升后的主累加结果。
每经过 mma_promotion_interval 条 MMA 指令，就把 accum_temp 逐元素加到 accum，然后清零 accum_temp，以缓解 FP8 长 K 累加的数值损失。
"""


# Helpers to parse args
# 参数解析辅助函数
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
    parser = argparse.ArgumentParser(
        description="FP8 GEMM with 2xAcc on Hopper (port of CUTLASS Example 54)."
    )

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
        default=(1, 2),
        help="Cluster shape (comma-separated)",
    )
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument(
        "--swizzle_size",
        type=int,
        default=1,
        help="Swizzling size in the unit of cluster for improving L2 cache hit rate",
    )
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument(
        "--raster_order",
        type=str,
        choices=["along_m", "along_n"],
        default="along_m",
        help="Rasterization order of clusters",
    )
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument(
        "--c_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
        help="Output dtype (Float16, Float32, or Float8E4M3FN)",
    )
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument(
        "--mma_promotion_interval",
        type=int,
        default=4,
        help="Number of MMA instructions between accumulator promotions (default: 4)",
    )
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument(
        "--scale_a",
        type=float,
        default=1.0,
        help="Scalar scale factor for A",
    )
    # 下面注册一个命令行参数；整个多行调用一起描述选项名、类型、默认值、可选范围和 help 文本。
    parser.add_argument(
        "--scale_b",
        type=float,
        default=1.0,
        help="Scalar scale factor for B",
    )
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


# 类 HopperFP8WarpSpecialized2xAccGemmKernel：FP8 2xAcc GEMM 封装类，固定 A/B 为 FP8 E4M3、累加为
# Float32，并实现两级累加策略。
class HopperFP8WarpSpecialized2xAccGemmKernel:
    """
    FP8 GEMM kernel with 2xAcc (double accumulation) for improved numerical accuracy.

    This kernel implements D = scale_a * scale_b * (A @ B) where A and B are FP8 E4M3FN
    tensors. The 2xAcc technique uses a temporary accumulator that is periodically promoted
    into the main accumulator to prevent precision loss from FP8 overflow.

    Based on the warp-specialized persistent tile scheduling pattern from dense_gemm_persistent.py,
    with the mainloop modified to implement the 2xAcc algorithm from CUTLASS's
    sm90_mma_tma_gmma_ss_warpspecialized_fp8.hpp.

    :param tile_shape_mn: Shape of the CTA tile (M,N)
    :type tile_shape_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]
    :param mma_promotion_interval: Number of MMA instructions between accumulator promotions
    :type mma_promotion_interval: int

    :note: Constraints:
        - Input types: FP8 E4M3FN only, k-major layout
        - Accumulation type: Float32
        - CTA tile M must be 64/128
        - CTA tile N must be 64/128/256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 4

    中文说明：
    这个类封装 FP8 E4M3FN 输入、Float32 累加的 2xAcc GEMM。它沿用 persistent/warp-specialized 框架，并在 mainloop 中周期性把临时 accumulator promote 到主 accumulator。
    """

    # 函数 HopperFP8WarpSpecialized2xAccGemmKernel.__init__：初始化 FP8 2xAcc kernel，包括 promotion
    # interval、scheduler 配置、warp group、寄存器预算和同步 barrier。 参数：self, tile_shape_mn, cluster_shape_mn,
    # swizzle_size, raster_along_m, mma_promotion_interval；返回：未显式标注。
    def __init__(
        self,
        tile_shape_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        swizzle_size: int,
        raster_along_m: bool,
        mma_promotion_interval: int = 4,
    ):
        self.acc_dtype = cutlass.Float32
        self.mma_promotion_interval = mma_promotion_interval # 解释1

        self.cluster_shape_mn = cluster_shape_mn # 解释2
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        self.mma_inst_shape_mn = None
        # K dimension is deferred in _setup_attributes
        # K 维 tile 大小会在 _setup_attributes 中根据 WGMMA 形状确定。
        self.tile_shape_mnk = (*tile_shape_mn, 1)
        # For large tile size, using two warp groups is preferred because using only one warp
        # group may result in register spill
        self.atom_layout_mnk = (
            (2, 1, 1)
            if self.tile_shape_mnk[0] > 64 and self.tile_shape_mnk[1] > 128
            else (1, 1, 1)
        )
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False
        self.tiled_mma = None

        self.occupancy = 1
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_warps_per_warp_group = 4
        self.num_threads_per_warp_group = self.num_warps_per_warp_group * 32
        self.threads_per_cta = (
            self.num_dma_warp_groups + self.num_mma_warp_groups
        ) * self.num_threads_per_warp_group
        self.load_warp_id = 0
        self.epi_store_warp_id = (
            self.num_dma_warp_groups * self.num_warps_per_warp_group
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

        self.num_mma_threads = (
            self.num_mma_warp_groups * self.num_threads_per_warp_group
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_mma_threads
        )

    # 函数 HopperFP8WarpSpecialized2xAccGemmKernel._setup_attributes：构造 FP8 WGMMA tiled_mma，验证
    # promotion interval，并生成 multicast、stage 和 SMEM layout。 参数：self；返回：未显式标注。
    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs.

        中文说明：
        根据 FP8 A/B 和输出 D 的 layout 派生 WGMMA tiled_mma、K tile、TMA multicast、epilogue tile、pipeline stage 和 SMEM layout，并验证 promotion interval。
        """

        # check the cta tile shape
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
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        # Validate that mma_promotion_interval is a multiple of num_k_blocks
        # 验证 promotion interval 必须是当前 K tile 内 WGMMA block 数的整数倍。
        # so the counter hits the interval exactly (promotion uses == not >=)
        # 这里 promotion 条件用 ==，所以计数器必须能精确命中 interval。
        num_k_blocks = mma_inst_tile_k
        if self.mma_promotion_interval % num_k_blocks != 0:
            # 遇到不支持的 dtype/layout/alignment 或运行环境时主动报错，避免继续生成非法 kernel。
            raise ValueError(
                f"mma_promotion_interval ({self.mma_promotion_interval}) must be a "
                f"multiple of num_k_blocks ({num_k_blocks})"
            )

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = self._sm90_compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
        )

        # Compute stage before compute smem layout
        # 下面根据 tile 大小、dtype 位宽和共享内存容量计算 pipeline stage 数；A/B stage 决定 mainloop 预取深度，epilogue stage
        # 决定写回缓冲数量。
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        # 下面一次性生成 A/B/epilogue 的 staged shared-memory layout；返回值按 A、B、C 写回顺序解包到实例属性，后续
        # TMA/WGMMA/epilogue 都会复用这些 layout。
        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
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

    # 函数 HopperFP8WarpSpecialized2xAccGemmKernel.__call__：FP8 host launch 包装：接收 A/B/D、scale tensor 和
    # max_active_clusters，创建 TMA 与 scheduler 后 launch。 参数：self, a, b, d, scale_a, scale_b,
    # max_active_clusters, stream；返回：未显式标注。
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        d: cute.Tensor,
        scale_a: cute.Tensor,
        scale_b: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Execute the FP8 GEMM with 2xAcc.

        :param a: Input tensor A (FP8 E4M3FN)
        :param b: Input tensor B (FP8 E4M3FN)
        :param d: Output tensor D
        :param scale_a: Scalar scale factor for A (1-element Float32 tensor)
        :param scale_b: Scalar scale factor for B (1-element Float32 tensor)
        :param max_active_clusters: Maximum number of active clusters
        :param stream: CUDA stream

        中文说明：
        host/JIT 入口：接收 A/B/D 和 scale_a/scale_b，创建 TMA atom/tensor 与 persistent scheduler 参数，定义 shared storage，然后 launch FP8 2xAcc device kernel。
        """

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = d.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(d)

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
        tma_atom_d, tma_tensor_d = self._make_tma_store_atoms_and_tensors(
            d,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        # 计算 kernel launch grid；persistent 版本还会同时生成 tile scheduler 参数。
        tile_sched_params, grid = self._compute_grid(
            d,
            self.tile_shape_mnk,
            self.cluster_shape_mn,
            self.swizzle_size,
            self.raster_along_m,
            max_active_clusters,
        )

        # 类 HopperFP8WarpSpecialized2xAccGemmKernel.__call__.SharedStorage：描述 FP8 persistent kernel
        # 的 shared memory：A/B pipeline buffer、D epilogue buffer 和 barrier。
        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sD: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.epi_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        # 下面是一个多行函数调用；用一段注释解释整个调用，参数行保持干净以便对照源码。
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_d,
            tma_tensor_d,
            scale_a,
            scale_b,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )
        return

    # GPU device kernel
    # 函数 HopperFP8WarpSpecialized2xAccGemmKernel.kernel：GPU FP8 kernel：DMA warp group 加载 A/B，MMA
    # warp group 用 accum_temp/accumulators 做 2xAcc，再 scale 并写回 D。 参数：self, tma_atom_a, mA_mkl,
    # tma_atom_b, mB_nkl, tma_atom_d, mD_mnl, scale_a, scale_b, tiled_mma, cta_layout_mnk,
    # a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged,
    # tile_sched_params；返回：未显式标注。
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD_mnl: cute.Tensor,
        scale_a: cute.Tensor,
        scale_b: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        """
        GPU device kernel performing FP8 GEMM with 2xAcc.

        The mainloop uses two accumulators:
        - accum_temp: temporary accumulator that WGMMA writes into
        - accumulators: main accumulator that collects promoted partial results

        Every mma_promotion_interval MMA instructions, accum_temp is promoted
        (element-wise added) into accumulators, then WGMMA is told to zero
        accum_temp on its next instruction.

        中文说明：
        device kernel 主体：DMA warp group 负责 TMA load；MMA warp group 执行 WGMMA 到 accum_temp，并按 promotion interval 合并到主 accumulator；epilogue 应用 scale 后写回 D。
        """

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch Tma desc
        # 按 warp 编号分配轻量控制工作，例如预取 TMA descriptor、发起 TMA copy 或执行 epilogue store。
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_d)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

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

        # Alloc and init AB full/empty + ACC full mbar (pipeline)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # mbar arrays
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # Threads/warps participating in this pipeline
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        # Each warp will contribute to the arrive count with the number of mcast size
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = (
            mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        # 创建 mainloop 的 TMA async pipeline，用 full/empty barrier 管理多 stage A/B shared-memory buffer。
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            defer_sync=True,
        )

        # Cluster arrive after barrier init
        # cluster 内 CTA 到达 pipeline 初始化同步点，确保 barrier 初始化过程可见。
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # Generate smem tensor A/B/D
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sD = storage.sD.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        # Local_tile partition global tensors
        # (bM, bK, RestM, RestK, RestL)
        # 从全局 tensor 中切出当前 tile/所有 tile 的局部视图，避免手写 M/N/K/L 索引计算。
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, RestN, RestK, RestL)
        # 从全局 tensor 中切出当前 tile/所有 tile 的局部视图，避免手写 M/N/K/L 索引计算。
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL)
        # 从全局 tensor 中切出当前 tile/所有 tile 的局部视图，避免手写 M/N/K/L 索引计算。
        gD_mnl = cute.local_tile(
            mD_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        # Partition shared tensor for TMA load A/B
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        # 把 global/shared tensor 按 TMA atom 和 CTA/cluster 坐标分区，得到 copy 指令需要的源和目的视图。
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        # 把 global/shared tensor 按 TMA atom 和 CTA/cluster 坐标分区，得到 copy 指令需要的源和目的视图。
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        # Partition global tensor for TiledMMA_A/B/C
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        mma_warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(
            mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups)
        )

        # Make fragments
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        tCgD = thr_mma.partition_C(gD_mnl)
        acc_shape = tCgD.shape[:3]
        # 创建寄存器 tensor，通常用于 accumulator、临时 accumulator 或 epilogue 类型转换缓冲。
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        # 2xAcc: create temporary accumulator
        # 创建寄存器 tensor，通常用于 accumulator、临时 accumulator 或 epilogue 类型转换缓冲。
        accum_temp = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # Cluster wait for barrier init
        # 等待 pipeline 初始化完成，避免在 barrier 未准备好时开始 producer/consumer 操作。
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        # 区分 DMA warp group 和 MMA warp group；persistent kernel 通过这个分支实现 warp specialization。
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        # =====================================================================
        # DMA warp group: TMA loads (identical to dense_gemm_persistent.py)
        # =====================================================================
        # 按 warp 编号分配轻量控制工作，例如预取 TMA descriptor、发起 TMA copy 或执行 epilogue store。
        if warp_idx == self.load_warp_id:
            # 创建 persistent tile scheduler，让当前 CTA/cluster 循环领取输出 tile，而不是只处理一个固定 tile。
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            mainloop_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )

            # persistent scheduler 主循环：当前 work tile 有效时持续处理输出 tile，完成后推进到下一个 tile。
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]

                mainloop_producer_state.reset_count()

                # 沿 K 维 tile 迭代 mainloop；每轮消费一个 K tile 的 A/B 数据并贡献一部分矩阵乘加。
                for k_tile in range(k_tile_cnt):
                    # Conditionally wait for AB buffer empty
                    # producer 等待目标 pipeline stage 变空，准备把新的 A/B tile 通过 TMA 搬入 shared memory。
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    # Slice to global/shared memref to current k_tile
                    tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                    tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                    tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                    tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                    # TMA load A/B
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
                    # producer 提交当前 pipeline stage；TMA async pipeline 中它主要推进状态语义。
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # producer 结束 pipeline，通知 consumer 不会再有新的 TMA stage。
            mainloop_pipeline.producer_tail(mainloop_producer_state)

        # =====================================================================
        # MMA warp group: 2xAcc mainloop + epilogue
        # =====================================================================
        # 区分 DMA warp group 和 MMA warp group；persistent kernel 通过这个分支实现 warp specialization。
        if not is_dma_warp_group:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            # 创建 persistent tile scheduler，让当前 CTA/cluster 循环领取输出 tile，而不是只处理一个固定 tile。
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            mainloop_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            mainloop_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )

            num_k_blocks = cute.size(tCrA, mode=[2])

            # Partition for epilogue
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
            thr_copy_r2s = tiled_copy_r2s.get_slice(
                tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group
            )
            # (t)hread-partition for (r)egister to (s)mem copy (tRS_)
            tRS_sD = thr_copy_r2s.partition_D(sD)
            # (R2S, R2S_M, R2S_N)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)

            # Allocate D registers.
            rD_shape = cute.shape(thr_copy_r2s.partition_S(sD))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            # 创建寄存器 tensor，通常用于 accumulator、临时 accumulator 或 epilogue 类型转换缓冲。
            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            # 创建寄存器 tensor，通常用于 accumulator、临时 accumulator 或 epilogue 类型转换缓冲。
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            k_pipe_mmas = 1
            prologue_mma_cnt = min(k_pipe_mmas, k_tile_cnt)

            # Initialize tma store pipeline
            tma_store_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_threads,
            )
            # 创建 epilogue 的 TMA store pipeline，用来协调 shared-to-global 异步写回。
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=tma_store_producer_group,
            )

            # Load scalar scale factors (all threads load the same values)
            # 读取 FP8 GEMM 的缩放因子，并合并成 epilogue 中要乘到 accumulator 上的 scale。
            scale_val = scale_a[0] * scale_b[0]

            # persistent scheduler 主循环：当前 work tile 有效时持续处理输出 tile，完成后推进到下一个 tile。
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gD_mnl_slice = gD_mnl[(None, None, *tile_coord_mnl)]

                # =============================================================
                # 2xAcc MAINLOOP
                # =============================================================
                mainloop_consumer_read_state.reset_count()
                # consumer 释放已经用完的 pipeline stage，让 producer 后续可以复用该 shared-memory buffer。
                mainloop_consumer_release_state.reset_count()
                accumulators.fill(0.0)

                # Start with ACCUMULATE=False so first GMMA zeros accum_temp
                # 下面是一个多行函数调用；用一段注释解释整个调用，参数行保持干净以便对照源码。
                tiled_mma.set(
                    cute.nvgpu.warpgroup.Field.ACCUMULATE, False
                )
                mma_count = 0

                cute.nvgpu.warpgroup.fence()

                # Prologue: first k_pipe_mmas k_tiles (no release)
                # 沿 K 维 tile 迭代 mainloop；每轮消费一个 K tile 的 A/B 数据并贡献一部分矩阵乘加。
                for k_tile in range(prologue_mma_cnt):
                    # Wait for TMA copies to complete
                    # consumer 等待当前 pipeline stage 的 TMA load 完成，确保 WGMMA 读取有效的 shared-memory 数据。
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                    # WGMMA into accum_temp
                    # 遍历当前 K tile 内的 WGMMA K-block；每个 block 发起一次 CuTe GEMM/WGMMA。
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        # 发起一次 CuTe GEMM/WGMMA，把当前 A/B fragment 累加到 accumulator。
                        cute.gemm(
                            tiled_mma,
                            accum_temp,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accum_temp,
                        )
                        # 下面是一个多行函数调用；用一段注释解释整个调用，参数行保持干净以便对照源码。
                        tiled_mma.set(
                            cute.nvgpu.warpgroup.Field.ACCUMULATE, True
                        )

                    # 提交当前 WGMMA group，让异步矩阵乘加进入执行队列。
                    cute.nvgpu.warpgroup.commit_group()

                    # 2xAcc: promote_if_needed
                    # 更新 FP8 2xAcc 的 MMA 计数器；到达 promotion interval 后会把 accum_temp 合入主 accumulator。
                    mma_count += num_k_blocks
                    # 检查 FP8 2xAcc 是否到达 promotion 条件；满足时等待 WGMMA 完成并把 accum_temp 累加到主 accumulator。
                    if mma_count == self.mma_promotion_interval:
                        # 等待 WGMMA group 完成；在读取 accumulator 或释放 buffer 前必须保证写入结束。
                        cute.nvgpu.warpgroup.wait_group(0)
                        # Element-wise promotion: accumulators += accum_temp
                        # 逐元素遍历寄存器 tensor；FP8 2xAcc 用它做 promotion 或 scale，示例中写得更直观。
                        for i in range(cute.size(accumulators)):
                            # 分配 accumulator 相关寄存器 tensor；普通累加写 accumulators，FP8 2xAcc 还会用
                            # accum_temp 做临时累加。
                            accumulators[i] = accumulators[i] + accum_temp[i]
                        mma_count = 0
                        # Signal WGMMA to zero accum_temp on next instruction
                        # 下面是一个多行函数调用；用一段注释解释整个调用，参数行保持干净以便对照源码。
                        tiled_mma.set(
                            cute.nvgpu.warpgroup.Field.ACCUMULATE, False
                        )

                    mainloop_consumer_read_state.advance()

                # Main loop: remaining k_tiles (with release)
                # 沿 K 维 tile 迭代 mainloop；每轮消费一个 K tile 的 A/B 数据并贡献一部分矩阵乘加。
                for k_tile in range(prologue_mma_cnt, k_tile_cnt):
                    # Wait for TMA copies to complete
                    # consumer 等待当前 pipeline stage 的 TMA load 完成，确保 WGMMA 读取有效的 shared-memory 数据。
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                    # WGMMA into accum_temp
                    # 遍历当前 K tile 内的 WGMMA K-block；每个 block 发起一次 CuTe GEMM/WGMMA。
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        # 发起一次 CuTe GEMM/WGMMA，把当前 A/B fragment 累加到 accumulator。
                        cute.gemm(
                            tiled_mma,
                            accum_temp,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accum_temp,
                        )
                        # 下面是一个多行函数调用；用一段注释解释整个调用，参数行保持干净以便对照源码。
                        tiled_mma.set(
                            cute.nvgpu.warpgroup.Field.ACCUMULATE, True
                        )

                    # 提交当前 WGMMA group，让异步矩阵乘加进入执行队列。
                    cute.nvgpu.warpgroup.commit_group()
                    # Wait on the wgmma barrier for WGMMA to complete
                    # 等待 WGMMA group 完成；在读取 accumulator 或释放 buffer 前必须保证写入结束。
                    cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)

                    # 2xAcc: promote_if_needed
                    # 更新 FP8 2xAcc 的 MMA 计数器；到达 promotion interval 后会把 accum_temp 合入主 accumulator。
                    mma_count += num_k_blocks
                    # 检查 FP8 2xAcc 是否到达 promotion 条件；满足时等待 WGMMA 完成并把 accum_temp 累加到主 accumulator。
                    if mma_count == self.mma_promotion_interval:
                        # Wait for all outstanding WGMMA writes to accum_temp
                        # before reading it (matches C++ warpgroup_wait<0>)
                        # 等待 WGMMA group 完成；在读取 accumulator 或释放 buffer 前必须保证写入结束。
                        cute.nvgpu.warpgroup.wait_group(0)
                        # Element-wise promotion: accumulators += accum_temp
                        # 逐元素遍历寄存器 tensor；FP8 2xAcc 用它做 promotion 或 scale，示例中写得更直观。
                        for i in range(cute.size(accumulators)):
                            # 分配 accumulator 相关寄存器 tensor；普通累加写 accumulators，FP8 2xAcc 还会用
                            # accum_temp 做临时累加。
                            accumulators[i] = accumulators[i] + accum_temp[i]
                        mma_count = 0
                        # Signal WGMMA to zero accum_temp on next instruction
                        # 下面是一个多行函数调用；用一段注释解释整个调用，参数行保持干净以便对照源码。
                        tiled_mma.set(
                            cute.nvgpu.warpgroup.Field.ACCUMULATE, False
                        )

                    # consumer 释放已经用完的 pipeline stage，让 producer 后续可以复用该 shared-memory buffer。
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    # consumer 释放已经用完的 pipeline stage，让 producer 后续可以复用该 shared-memory buffer。
                    mainloop_consumer_release_state.advance()
                    mainloop_consumer_read_state.advance()

                # 等待 WGMMA group 完成；在读取 accumulator 或释放 buffer 前必须保证写入结束。
                cute.nvgpu.warpgroup.wait_group(0)

                # 2xAcc: promote_residue - promote any remaining partial results
                # 检查 FP8 2xAcc 是否到达 promotion 条件；满足时等待 WGMMA 完成并把 accum_temp 累加到主 accumulator。
                if mma_count > 0:
                    # 逐元素遍历寄存器 tensor；FP8 2xAcc 用它做 promotion 或 scale，示例中写得更直观。
                    for i in range(cute.size(accumulators)):
                        # 分配 accumulator 相关寄存器 tensor；普通累加写 accumulators，FP8 2xAcc 还会用 accum_temp
                        # 做临时累加。
                        accumulators[i] = accumulators[i] + accum_temp[i]

                # Release remaining pipeline stages from prologue
                # 沿 K 维 tile 迭代 mainloop；每轮消费一个 K tile 的 A/B 数据并贡献一部分矩阵乘加。
                for k_tile in range(prologue_mma_cnt):
                    # consumer 释放已经用完的 pipeline stage，让 producer 后续可以复用该 shared-memory buffer。
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    # consumer 释放已经用完的 pipeline stage，让 producer 后续可以复用该 shared-memory buffer。
                    mainloop_consumer_release_state.advance()

                # =============================================================
                # Epilogue: apply scaling, then R2S -> S2G
                # =============================================================

                # Apply scale_a * scale_b to accumulators
                # 逐元素遍历寄存器 tensor；FP8 2xAcc 用它做 promotion 或 scale，示例中写得更直观。
                for i in range(cute.size(accumulators)):
                    # 分配 accumulator 相关寄存器 tensor；普通累加写 accumulators，FP8 2xAcc 还会用 accum_temp 做临时累加。
                    accumulators[i] = accumulators[i] * scale_val

                tCgD_for_tma_partition = cute.zipped_divide(gD_mnl_slice, self.epi_tile)

                # thread(b)lock-partition for (s)mem to (g)mem copy (bSG_)
                # 把 global/shared tensor 按 TMA atom 和 CTA/cluster 坐标分区，得到 copy 指令需要的源和目的视图。
                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_d,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sD, 0, 2),
                    tCgD_for_tma_partition,
                )

                epi_tile_num = cute.size(tCgD_for_tma_partition, mode=[1])
                epi_tile_shape = tCgD_for_tma_partition.shape[1]
                epi_tile_layout = cute.make_layout(
                    epi_tile_shape, stride=(epi_tile_shape[1], 1)
                )

                num_prev_epi_tiles = tile_sched.num_tiles_executed * epi_tile_num
                # 遍历 epilogue 子 tile，把 accumulator 分块转换、写入 shared memory，再通过 TMA store 写回输出。
                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    # Copy from accumulators to D registers
                    # 遍历 epilogue 子 tile，把 accumulator 分块转换、写入 shared memory，再通过 TMA store 写回输出。
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

                    # Type conversion (acc_dtype -> c_dtype)
                    acc_vec = tRS_rD.load()
                    tRS_rD_out.store(acc_vec.to(self.c_dtype))

                    # Copy from D registers to shared memory
                    epi_buffer = (num_prev_epi_tiles + epi_idx) % cute.size(
                        tRS_sD, mode=[3]
                    )
                    # 执行 CuTe copy；根据上下文可能是 TMA load、TMA store 或寄存器到 shared memory 的 copy。
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )

                    # 下面是一个多行函数调用；用一段注释解释整个调用，参数行保持干净以便对照源码。
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    self.epilog_sync_barrier.arrive_and_wait()

                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    # Copy from shared memory to global memory
                    # 按 warp 编号分配轻量控制工作，例如预取 TMA descriptor、发起 TMA copy 或执行 epilogue store。
                    if warp_idx == self.epi_store_warp_id:
                        # 执行 CuTe copy；根据上下文可能是 TMA load、TMA store 或寄存器到 shared memory 的 copy。
                        cute.copy(
                            tma_atom_d,
                            bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, gmem_coord)],
                        )
                        # producer 提交当前 pipeline stage；TMA async pipeline 中它主要推进状态语义。
                        tma_store_pipeline.producer_commit()
                        # producer 等待目标 pipeline stage 变空，准备把新的 A/B tile 通过 TMA 搬入 shared memory。
                        tma_store_pipeline.producer_acquire()

                    self.epilog_sync_barrier.arrive_and_wait()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # producer 结束 pipeline，通知 consumer 不会再有新的 TMA stage。
            tma_store_pipeline.producer_tail()

    # 函数 HopperFP8WarpSpecialized2xAccGemmKernel._compute_stages：估算 FP8 kernel 的 A/B pipeline stage
    # 和 D epilogue stage。 参数：tile_shape_mnk, a_dtype, b_dtype, epi_tile, c_dtype, smem_capacity,
    # occupancy；返回：tuple[int, int]。
    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: tuple[int, int],
        c_dtype: type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        中文说明：
        按 tile、dtype、epilogue tile 和 SMEM 容量估算 A/B mainloop stage 与 epilogue stage，给 FP8 pipeline 分配 shared memory。
        """
        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_stage = 4
        epi_bytes = c_bytes_per_stage * epi_stage

        mbar_helpers_bytes = 1024

        ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes)
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    # 函数 HopperFP8WarpSpecialized2xAccGemmKernel._sm90_compute_tile_shape_or_override：选择 FP8
    # epilogue tile，8-bit 输出倾向更大的 N 子 tile。 参数：tile_shape_mnk, element_type, is_cooperative,
    # epi_tile_override；返回：tuple[int, int]。
    @staticmethod
    def _sm90_compute_tile_shape_or_override(
        tile_shape_mnk: tuple[int, int, int],
        element_type: type[cutlass.Numeric],
        is_cooperative: bool = False,
        epi_tile_override: Optional[tuple[int, int]] = None,
    ) -> tuple[int, int]:
        """Compute the epilogue tile shape or use override if provided.

        中文说明：
        计算 epilogue 写回 D 的子 tile 形状；输出 dtype 较窄时可使用更大的 N 方向宽度以提高写回效率。
        """
        if epi_tile_override is not None:
            return epi_tile_override
        if is_cooperative:
            tile_m = min(128, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(32, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)
        else:
            n_perf = 64 if element_type.width == 8 else 32
            tile_m = min(64, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(n_perf, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)

    # 函数 HopperFP8WarpSpecialized2xAccGemmKernel._make_smem_layouts：构造 FP8 A/B 和 D 的 staged shared-
    # memory layout。 参数：tile_shape_mnk, epi_tile, a_dtype, a_layout, b_dtype, b_layout, ab_stage,
    # c_dtype, c_layout, epi_stage；返回：tuple[cute.ComposedLayout, cute.ComposedLayout,
    # cute.ComposedLayout]。
    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: utils.LayoutEnum,
        b_dtype: type[cutlass.Numeric],
        b_layout: utils.LayoutEnum,
        ab_stage: int,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        """Create shared memory layouts for A, B, and D tensors.

        中文说明：
        创建 A/B/D 的 staged SMEM layout。A/B 用于 TMA load 与 WGMMA 读取，D 用于 epilogue 暂存和 TMA store。
        """
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))

        a_is_k_major = (
            a_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        )
        b_is_k_major = (
            b_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        )
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                a_layout,
                a_dtype,
                a_major_mode_size,
            ),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))

        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                b_layout,
                b_dtype,
                b_major_mode_size,
            ),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        c_smem_shape = epi_tile
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                c_layout,
                c_dtype,
                c_major_mode_size,
            ),
            c_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(c_smem_shape, epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
        )

        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    # 函数 HopperFP8WarpSpecialized2xAccGemmKernel._compute_grid：构造 persistent scheduler 参数，并计算受
    # max_active_clusters 限制的 launch grid。 参数：d, tile_shape_mnk, cluster_shape_mn, swizzle_size,
    # raster_along_m, max_active_clusters；返回：tuple[int, int, int]。
    @staticmethod
    def _compute_grid(
        d: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        swizzle_size: int,
        raster_along_m: bool,
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[int, int, int]:
        """Compute grid shape for the output tensor D.

        中文说明：
        根据 D 的 tile 划分、cluster shape 和 max_active_clusters 生成 persistent scheduler 参数与 launch grid。
        """
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gd = cute.zipped_divide(d, tiler=c_shape)
        num_ctas_mnl = gd[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl,
            cluster_shape_mnl,
            swizzle_size,
            raster_along_m,
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    # 函数 HopperFP8WarpSpecialized2xAccGemmKernel._make_tma_store_atoms_and_tensors：创建 D 的 TMA
    # shared-to-global store atom 和 tensor 视图。 参数：tensor_d, epi_smem_layout_staged,
    # epi_tile；返回：tuple[cute.CopyAtom, cute.Tensor]。
    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_d: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for D tensor storage.

        中文说明：
        创建 D 的 shared-to-global TMA store atom 和 tensor 视图，用于 epilogue 写回输出。
        """
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        # 下面是一次多返回值解包：把右侧计算结果拆成 (tma_atom_d, tma_tensor_d)，多行参数保持原代码结构，不逐行解释。
        tma_atom_d, tma_tensor_d = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_d,
            epi_smem_layout,
            epi_tile,
        )

        return tma_atom_d, tma_tensor_d

    # 函数 HopperFP8WarpSpecialized2xAccGemmKernel._make_tma_atoms_and_tensors：创建 FP8 A/B 的 TMA
    # global-to-shared load atom；必要时使用 multicast。 参数：tensor, smem_layout_staged, smem_tile,
    # mcast_dim；返回：tuple[cute.CopyAtom, cute.Tensor]。
    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        中文说明：
        创建 A/B 的 global-to-shared TMA load atom 和 tensor 视图；cluster 维度大于 1 时启用 multicast。
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
        return tma_atom, tma_tensor


# 函数 run：FP8 2xAcc host 示例入口：构造 FP8 tensor 和 scale tensor，编译、校验并 benchmark。 参数：mnkl, c_dtype,
# tile_shape_mn, cluster_shape_mn, swizzle_size, raster_along_m, mma_promotion_interval,
# scale_a_val, scale_b_val, tolerance, warmup_iterations, iterations, skip_ref_check, use_cold_l2,
# **kwargs；返回：未显式标注。
def run(
    mnkl: Tuple[int, int, int, int],
    c_dtype: Type[cutlass.Numeric],
    tile_shape_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    swizzle_size: int = 1,
    raster_along_m: bool = True,
    mma_promotion_interval: int = 4,
    scale_a_val: float = 1.0,
    scale_b_val: float = 1.0,
    tolerance: float = 1e-01,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    **kwargs,
):
    """
    Prepare FP8 A/B tensors, launch GPU kernel with 2xAcc, and reference checking.

    :param mnkl: Problem size (M, N, K, L)
    :param c_dtype: Data type for output tensor D
    :param tile_shape_mn: CTA tile shape (M, N)
    :param cluster_shape_mn: Cluster shape (M, N)
    :param mma_promotion_interval: MMA instructions between accumulator promotions
    :param scale_a_val: Scalar scale factor for A
    :param scale_b_val: Scalar scale factor for B
    :param tolerance: Tolerance value for reference validation
    :param warmup_iterations: Number of warmup iterations
    :param iterations: Number of benchmark iterations
    :param skip_ref_check: Whether to skip reference validation
    :param use_cold_l2: Whether to use cold L2 cache strategy
    :return: Execution time in microseconds

    中文说明：
    完整 host 示例入口：创建 FP8 A/B 和输出 D，准备 scale tensor，编译并运行 kernel，执行可选参考校验和 benchmark。
    """
    import torch
    import cutlass.torch as cutlass_torch

    a_dtype = cutlass.Float8E4M3FN
    b_dtype = cutlass.Float8E4M3FN
    acc_dtype = cutlass.Float32

    print("Running Hopper FP8 Dense GEMM with 2xAcc:")
    print(f"mnkl: {mnkl}")
    # 下面是一个多行函数调用；用一段注释解释整个调用，参数行保持干净以便对照源码。
    print(
        f"A dtype: {a_dtype}, B dtype: {b_dtype}, D dtype: {c_dtype}, Acc dtype: {acc_dtype}"
    )
    print(f"Tile Shape: {tile_shape_mn}, Cluster Shape: {cluster_shape_mn}")
    print(f"MMA promotion interval: {mma_promotion_interval}")
    print(f"scale_a: {scale_a_val}, scale_b: {scale_b_val}")
    # 下面是一个多行函数调用；用一段注释解释整个调用，参数行保持干净以便对照源码。
    print(
        f"Swizzle size: {swizzle_size}, Raster order:",
        "along_m" if raster_along_m else "along_n",
    )
    print(f"Tolerance: {tolerance}")

    # Unpack parameters
    m, n, k, l = mnkl

    # 运行前合法性检查：不支持的 dtype/layout/alignment 或无 GPU 环境会提前报错。
    if not torch.cuda.is_available():
        # 遇到不支持的 dtype/layout/alignment 或运行环境时主动报错，避免继续生成非法 kernel。
        raise RuntimeError("GPU is required to run this example!")

    # Validate alignment
    num_contiguous_elements = 16 * 8 // a_dtype.width  # 16 for FP8
    if k % num_contiguous_elements != 0:
        # 遇到不支持的 dtype/layout/alignment 或运行环境时主动报错，避免继续生成非法 kernel。
        raise ValueError(
            f"K dimension ({k}) must be aligned to {num_contiguous_elements} elements for FP8"
        )

    # Create FP8 input tensors (k-major)
    # 下面是一个关键 helper/API 调用；它生成后续 kernel 配置、tensor 视图、pipeline 状态或 benchmark 工作区。
    a_torch_cpu = cutlass_torch.matrix(l, m, k, False, a_dtype)  # k-major
    # 下面是一个关键 helper/API 调用；它生成后续 kernel 配置、tensor 视图、pipeline 状态或 benchmark 工作区。
    b_torch_cpu = cutlass_torch.matrix(l, n, k, False, b_dtype)  # k-major
    # 下面是一个关键 helper/API 调用；它生成后续 kernel 配置、tensor 视图、pipeline 状态或 benchmark 工作区。
    d_torch_cpu = cutlass_torch.matrix(l, m, n, False, c_dtype)  # n-major

    # 下面是一个关键 helper/API 调用；它生成后续 kernel 配置、tensor 视图、pipeline 状态或 benchmark 工作区。
    a_tensor, _ = cutlass_torch.cute_tensor_like(
        a_torch_cpu, a_dtype, is_dynamic_layout=True, assumed_align=16
    )
    # 下面是一个关键 helper/API 调用；它生成后续 kernel 配置、tensor 视图、pipeline 状态或 benchmark 工作区。
    b_tensor, _ = cutlass_torch.cute_tensor_like(
        b_torch_cpu, b_dtype, is_dynamic_layout=True, assumed_align=16
    )
    # 下面是一个关键 helper/API 调用；它生成后续 kernel 配置、tensor 视图、pipeline 状态或 benchmark 工作区。
    d_tensor, d_torch_gpu = cutlass_torch.cute_tensor_like(
        d_torch_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16
    )

    # Create scalar scale tensors on GPU
    scale_a_torch = torch.tensor([scale_a_val], dtype=torch.float32, device="cuda")
    scale_b_torch = torch.tensor([scale_b_val], dtype=torch.float32, device="cuda")
    # 下面是一个关键 helper/API 调用；它生成后续 kernel 配置、tensor 视图、pipeline 状态或 benchmark 工作区。
    scale_a_tensor, _ = cutlass_torch.cute_tensor_like(
        scale_a_torch, cutlass.Float32, is_dynamic_layout=True, assumed_align=16
    )
    # 下面是一个关键 helper/API 调用；它生成后续 kernel 配置、tensor 视图、pipeline 状态或 benchmark 工作区。
    scale_b_tensor, _ = cutlass_torch.cute_tensor_like(
        scale_b_torch, cutlass.Float32, is_dynamic_layout=True, assumed_align=16
    )

    gemm = HopperFP8WarpSpecialized2xAccGemmKernel(
        tile_shape_mn, cluster_shape_mn, swizzle_size, raster_along_m,
        mma_promotion_interval,
    )

    # Compute max active clusters on current device
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    # Compile gemm kernel
    # 触发 CuTe DSL JIT 编译，把 Python kernel 描述和示例参数 specialize 成可 launch 的 CUDA kernel。
    compiled_gemm = cute.compile(
        gemm, a_tensor, b_tensor, d_tensor, scale_a_tensor, scale_b_tensor,
        max_active_clusters, stream
    )

    # 根据命令行选项决定是否跳过参考校验；跳过会更快，但不会验证输出正确性。
    if not skip_ref_check:
        compiled_gemm(a_tensor, b_tensor, d_tensor, scale_a_tensor, scale_b_tensor, stream)
        torch.cuda.synchronize()

        # Compute reference result: D = scale_a * scale_b * (A @ B)
        # 用 PyTorch 计算参考 GEMM 结果，用于和自定义 kernel 输出做数值校验。
        ref = torch.einsum(
            "mkl,nkl->mnl",
            a_torch_cpu.to(dtype=torch.float32),
            b_torch_cpu.to(dtype=torch.float32),
        )
        ref = ref * scale_a_val * scale_b_val

        # Convert ref to c_dtype
        # 下面是一个关键 helper/API 调用；它生成后续 kernel 配置、tensor 视图、pipeline 状态或 benchmark 工作区。
        _, ref_torch_gpu = cutlass_torch.cute_tensor_like(
            ref, c_dtype, is_dynamic_layout=True, assumed_align=16
        )
        ref_d = ref_torch_gpu.cpu()

        # Assert close results
        # 比较 kernel 输出和 PyTorch 参考结果，容差内则认为数值正确。
        torch.testing.assert_close(
            d_torch_gpu.cpu().float(), ref_d.float(), atol=tolerance, rtol=1e-03
        )

    # 函数 run.generate_tensors：FP8 benchmark 的 workspace 生成器，复用 scale tensor 并重建 A/B/D workspace。
    # 参数：无；返回：未显式标注。
    def generate_tensors():
        # 下面是一个关键 helper/API 调用；它生成后续 kernel 配置、tensor 视图、pipeline 状态或 benchmark 工作区。
        a_tensor_workspace, _ = cutlass_torch.cute_tensor_like(
            a_torch_cpu, a_dtype, is_dynamic_layout=True, assumed_align=16
        )
        # 下面是一个关键 helper/API 调用；它生成后续 kernel 配置、tensor 视图、pipeline 状态或 benchmark 工作区。
        b_tensor_workspace, _ = cutlass_torch.cute_tensor_like(
            b_torch_cpu, b_dtype, is_dynamic_layout=True, assumed_align=16
        )
        # 下面是一个关键 helper/API 调用；它生成后续 kernel 配置、tensor 视图、pipeline 状态或 benchmark 工作区。
        d_tensor_workspace, _ = cutlass_torch.cute_tensor_like(
            d_torch_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16
        )
        return testing.JitArguments(
            a_tensor_workspace, b_tensor_workspace, d_tensor_workspace,
            scale_a_tensor, scale_b_tensor, stream
        )

    workspace_count = 1
    # 根据选项启用 cold L2 benchmark 策略，通过轮换 workspace 降低缓存复用影响。
    if use_cold_l2:
        one_workspace_bytes = (
            a_torch_cpu.numel() * a_torch_cpu.element_size()
            + b_torch_cpu.numel() * b_torch_cpu.element_size()
            + d_torch_cpu.numel() * d_torch_cpu.element_size()
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
        args.c_dtype,
        args.tile_shape_mn,
        args.cluster_shape_mn,
        args.swizzle_size,
        True if args.raster_order == "along_m" else False,
        args.mma_promotion_interval,
        args.scale_a,
        args.scale_b,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )
    print("PASS")
