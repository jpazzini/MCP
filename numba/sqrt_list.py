from numba import cuda
import numpy as np

@cuda.jit
def kernel(x, max_index, out):
    index = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (index < max_index):
        out[index] = x[index]**0.5

def main():
    # create list on host
    n_items = 4096
    h_list = np.arange(n_items,dtype=np.float32)

    # copy list to device
    d_list = cuda.to_device(h_list)

    # create array on device like input
    d_res_list = cuda.device_array_like(d_list)

    # evaluate the number of kernels to lauch
    threads_per_block = 10
    blocks_per_grid = (n_items + threads_per_block + 1)//threads_per_block

    # launch kernel on device
    kernel[blocks_per_grid, threads_per_block](d_list, n_items, d_res_list)

    # retrieve data from device
    res_list = d_res_list.copy_to_host()

    print(res_list)

if __name__ == "__main__":
    main()
