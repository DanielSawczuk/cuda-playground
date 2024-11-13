import os
from pathlib import Path

import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import math

from pycuda.compiler import SourceModule

kernel_file = Path(".") / "src" / "kernels" / "gemm_wmma.cu"
include_dir = Path(".") / "include" / "tools"

with kernel_file.open() as src_fp:
    # pycuda wraps the whole kernel code (including includes) with `extern "C"` by default,
    # it causes problems if included headers contain templated functions (e.g. mma.h),
    # that's why `not_extern_c` needs to be set to `True`,
    # and to avoid dealing with C++ mangled function names,
    # `extern "C"` is added manually only to the __global__ function
    mod = SourceModule(
        src_fp.read(),
        no_extern_c=True,
        include_dirs=[str(include_dir.resolve())],
    )  # , options=["-DDEBUG"])

multiply_them = mod.get_function("gemm")

M = 64
K = 128
N = 48

a = np.random.randn(M, K).astype(np.float16)
b = np.random.randn(K, N).astype(np.float16)
c = np.zeros((M, N), dtype=np.float32)

golden_c = np.matmul(a, b) + c

multiply_them(
    drv.In(a),
    drv.In(b),
    drv.InOut(c),
    np.int32(M),
    np.int32(K),
    np.int32(N),
    block=(math.ceil(M / 16) * 32, math.ceil(N / 16), 1),
    grid=(1, 1, 1),
)

rtol = 1e-2
atol = 1e-8
passed = np.allclose(c, golden_c, rtol=rtol, atol=atol)
if not passed:
    err = (atol + rtol * abs(golden_c)) - abs(c - golden_c)
    max_rerr = np.argmax(err)
    print(
        f"Fail, max error: {c.ravel()[max_rerr]} != {golden_c.ravel()[max_rerr]}; using: rtol = {rtol}, atol = {atol}"
    )
else:
    print("Pass")
