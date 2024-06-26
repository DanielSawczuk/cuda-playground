import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
with open("mul.cu") as src_fp:
  mod = SourceModule(src_fp.read())

multiply_them = mod.get_function("multiply_them")

a_size = 1024 * 38 * 2
b_size = 1024 * 38 * 2

a = numpy.random.randn(a_size).astype(numpy.float32)
b = numpy.random.randn(b_size).astype(numpy.float32)

dest = numpy.zeros_like(a)
multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(1024,1,1), grid=(38 * 2,1))

print(dest-a*b)
assert numpy.allclose(dest, a*b)
print(dest)
