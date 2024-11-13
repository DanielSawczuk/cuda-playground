# cuda-playground

## Set up CUDA environment
```
export CUDA_TOOLKIT_PATH=/usr/local/cuda-12.3/
export PATH=$CUDA_TOOLKIT_PATH/bin:$PATH
export CPATH=$CUDA_TOOLKIT_PATH/include:$CPATH
```

## Python (pycuda) example
```
conda env create --file environment.yml
conda activate cuda
python example.py
```

## Cpp example
#### Compile
```
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86-virtual && make -j
```

#### Run
```
src/gemm_example src/kernels/gemm_fma.ptx
```

## Misc
#### Set up pre-coomit hook for clang-format and black formatting
```
ln tools/hooks/format.hook .git/hooks/pre-commit
```

#### Dumping PTX
```
nvcc --ptx -arch sm_86 -Iinclude/tools src/kernels/gemm_fma.cu
```


#### Profiling
```
export CUDA_NCU_DIR=/usr/local/NVIDIA-Nsight-Compute/
```
Python
```
sudo -E bash -c "PATH=$CUDA_TOOLKIT_PATH/bin:$PATH $CUDA_NCU_DIR/ncu --config-file off --export prof_log --set full --force-overwrite python example.py"
```
Cpp
```
sudo -E bash -c "PATH=$CUDA_TOOLKIT_PATH/bin:$PATH $CUDA_NCU_DIR/ncu --config-file off --export prof_log --set full --force-overwrite build/src/gemm_example build/src/kernels/gemm_fma.ptx"
```
