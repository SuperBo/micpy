# Micpy

micpy is a modified version of [numpy](https://github.com/numpy/numpy) in order to run on Intel Xeon Phi Knights Corner coprocessors.

## Compile and Installation
In order to compile micpy, you must have Intel C Compiler and Intel MKL installed on the host.
```Shell
python setup.py build
python setup.py install
```
## Usage
Below is an example of how to use micpy. Its usage is somewhat similar to original numpy.
```python
import micpy

# Create two arrays that reside on MIC device memory
array_a = micpy.zeros((100,100), dtype=micpy.float32)
array_b = micpy.ones((100,100), dtype=micpy.float32)

# Some calculations
array_c = array_a + array_b
array_d = array_a * array_b

# Transfer data between host memory and MIC device memory
c = micpy.to_cpu(array_c)
c = array_c.to_cpu()
array_e = micpy.to_mic(c)
```

## Current support API
TODO: update later

## Benchmarks
TODO: update later
