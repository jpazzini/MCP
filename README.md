## Material for the Course: _Modern Computing for Physics_ 
### M.Sc. in Physics of Data — University of Padova — Academic Year 2025-2026

#### CUDA-C

- Hello World in CUDA
- Scalar Sum 
- Vector Sum
- Elementwise Matrix Manipulation
- Matrix Multiplication
- Stencil Operations
- Reduction Operations

#### Numba

- `jit`
- `vectorize`
- `guvectorize`

#### CUDA in Python with Numba and CuPy

- `(gu)vectorize` with CUDA Backend
- `cuda.jit`
- `cupy`

*All code has been tested targeting the NVIDIA Jetson Nano 2GB Dev Kit.*

#### Warning

For more time-consuming computations on the NVIDIA Jetson Nano, it is recommended to verify that the system timeout, which is set by default at 5 seconds, is disabled. 
To do this, run the script provided under the `MCP` folder as `sudo`:

```bash
sudo ./disable_timeout.sh
```
