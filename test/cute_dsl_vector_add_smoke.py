#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

"""Small CuTe DSL smoke test that runs a CUDA vector-add kernel."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Make CUTLASS helper modules importable without shadowing an installed CuTe DSL.
PYTHON_HELPERS = str(REPO_ROOT / "python")
if PYTHON_HELPERS not in sys.path:
    sys.path.insert(0, PYTHON_HELPERS)

# Prefer this checkout only when it contains the generated CuTe DSL runtime.
LOCAL_CUTE_DSL = REPO_ROOT / "python" / "CuTeDSL"
if (LOCAL_CUTE_DSL / "cutlass" / "_mlir").is_dir():
    local_cute_dsl = str(LOCAL_CUTE_DSL)
    if local_cute_dsl not in sys.path:
        sys.path.insert(0, local_cute_dsl)

# Keep generated files out of cutlass/test unless the caller explicitly opts out.
os.environ.setdefault("CUTE_DSL_CACHE_DIR", "/tmp/cutlass_cute_dsl_smoke_cache")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", "/tmp/cutlass_cute_dsl_smoke_dump")

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
except ModuleNotFoundError as exc:
    if exc.name == "cutlass" or exc.name.startswith("cutlass."):
        raise RuntimeError(
            "CuTe DSL is not importable. Install/build the CuTe DSL Python package "
            "first, or run this script in an environment where `import cutlass.cute` "
            "works."
        ) from exc
    raise


@cute.kernel
def vector_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    cC: cute.Tensor,
    shape: cute.Shape,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    block_coord = ((None, None), bidx)
    blkA = gA[block_coord]
    blkB = gB[block_coord]
    blkC = gC[block_coord]
    blkCrd = cC[block_coord]

    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)

    tiled_copy_A = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_B = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_C = cute.make_tiled_copy_tv(copy_atom_store, thr_layout, val_layout)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    thr_copy_C = tiled_copy_C.get_slice(tidx)

    thrA = thr_copy_A.partition_S(blkA)
    thrB = thr_copy_B.partition_S(blkB)
    thrC = thr_copy_C.partition_S(blkC)
    thrCrd = thr_copy_C.partition_S(blkCrd)

    frgA = cute.make_rmem_tensor_like(thrA)
    frgB = cute.make_rmem_tensor_like(thrB)
    frgC = cute.make_rmem_tensor_like(thrC)
    frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)

    for i in range(0, cute.size(frgPred), 1):
        frgPred[i] = cute.elem_less(thrCrd[i], shape)

    cute.copy(copy_atom_load, thrA, frgA, pred=frgPred)
    cute.copy(copy_atom_load, thrB, frgB, pred=frgPred)
    frgC.store(frgA.load() + frgB.load())
    cute.copy(copy_atom_store, frgC, thrC, pred=frgPred)


@cute.jit
def vector_add(mA, mB, mC, copy_bits: cutlass.Constexpr = 128):
    dtype = mA.element_type
    vector_size = copy_bits // dtype.width

    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    cC = cute.zipped_divide(cute.make_identity_tensor(mC.shape), tiler=tiler_mn)

    vector_add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=8, help="Matrix rows.")
    parser.add_argument("--n", type=int, default=17, help="Matrix columns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.m <= 0 or args.n <= 0:
        raise ValueError("--m and --n must be positive")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to run this CuTe DSL smoke test.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    a = torch.arange(args.m * args.n, device=device, dtype=torch.float32).reshape(
        args.m, args.n
    )
    b = torch.randn(args.m, args.n, device=device, dtype=torch.float32)
    c = torch.empty_like(a)

    a_cute = from_dlpack(a).mark_layout_dynamic()
    b_cute = from_dlpack(b).mark_layout_dynamic()
    c_cute = from_dlpack(c).mark_layout_dynamic()

    compiled = cute.compile(vector_add, a_cute, b_cute, c_cute)
    compiled(a_cute, b_cute, c_cute)
    torch.cuda.synchronize()

    torch.testing.assert_close(c, a + b)
    print(
        f"PASS: CuTe DSL vector_add ran on {torch.cuda.get_device_name()} "
        f"for shape ({args.m}, {args.n})."
    )


if __name__ == "__main__":
    main()
