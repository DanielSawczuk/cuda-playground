# cuda-playground

```
export CPATH=/usr/local/cuda-12.3/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=/usr/local/cuda-12.3/lib64:/usr/local/cuda-12.3/targets/x86_64-linux/lib/stubs/:$LIBRARY_PATH
export PATH=/usr/local/cuda-12.3/bin:$PATH
conda env create --file environment.yml
conda activate cuda
python example.py
```

Profiling
```
sudo -E bash -c "PATH=/usr/local/cuda-12.3/bin:$PATH /opt/nvidia/nsight-compute/2023.3.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export prof_log --force-overwrite /home/${USER}/miniforge3/envs/cuda/bin/python example.py"
```

Dump PTX
```
nvcc --ptx -arch sm_86 -I/home/sdaniel/miniforge3/envs/cuda/lib/python3.12/site-packages/pycuda/cuda matmul.cu 
```