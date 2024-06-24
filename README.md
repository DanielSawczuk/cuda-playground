# cuda-playground

```
export CPATH=/usr/local/cuda-12.3/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=/usr/local/cuda-12.3/lib64:/usr/local/cuda-12.3/targets/x86_64-linux/lib/stubs/:$LIBRARY_PATH
export PATH=/usr/local/cuda-12.3/bin:$PATH
conda env create --file environment.yml
conda activate cuda
python example.py
```