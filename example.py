import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import math

from pycuda.compiler import SourceModule

with open("matmul.cu") as src_fp:
    # wmma functions are templated so to use them we need `no_extern_c=True`
    # (pycuda wraps functions with `extern C` by defualt)
    mod = SourceModule(src_fp.read(), no_extern_c=True)  # , options=["-DDEBUG"])

# because of `no_extern_c=True` we need C++ mangled function name
# (tip: compile kernel to PTX to check how it should look like)
multiply_them = mod.get_function("_Z6matmulP6__halfS0_Pfiii")

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
