#include <stdio.h>

#define N 1024                  // Length of input vector
#define KERNEL_RADIUS 3         // Radius of the smoothing kernel (kernel size = 2 * KERNEL_RADIUS + 1)
#define THREADS_PER_BLOCK 256   // Number of threads per block

// CUDA kernel for performing naive 1D convolution
__global__ void convolve1D(float* input, float* output, float* kernel, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float result = 0.0;

    if (i < n) {
        // Apply convolution with the 1D kernel
        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
            int idx = i + j;

            // Handle boundaries: zero-padding
            if (idx >= 0 && idx < n) {
                result += input[idx] * kernel[KERNEL_RADIUS + j];
            }
        }
        output[i] = result;
    }
}

// CUDA kernel for performing 1D convolution using shared memory
__global__ void convolve1D_sharedMemory(float* input, float* output, float* kernel, int n) {
    // Shared memory size for a block, with padding for the kernel radius on both sides
    __shared__ float shared_mem[THREADS_PER_BLOCK + 2 * KERNEL_RADIUS];

    int tid = threadIdx.x;                   // Local thread index within the block
    int gi = blockIdx.x * blockDim.x + tid;  // Global index of the data element
    int li = tid + KERNEL_RADIUS;            // Local index of the shared memory array

    // Load input data into shared memory
    if (gi < n) {
        shared_mem[li] = input[gi];
    } else {
        shared_mem[li] = 0.0f;  // Zero-padding for out-of-bounds
    }

    // Load halo elements (padding) into shared memory for the stencil boundaries
    if (tid < KERNEL_RADIUS) {
        // Left halo
        if (gi >= KERNEL_RADIUS) {
            shared_mem[tid] = input[gi - KERNEL_RADIUS];
        } else {
            shared_mem[tid] = 0.0;  // Zero-padding at the left boundary
        }

        // Right halo
        if (gi < n - blockDim.x) {
            shared_mem[li + blockDim.x] = input[gi + blockDim.x];
        } else {
            shared_mem[li + blockDim.x] = 0.0;  // Zero-padding at the right boundary
        }
    }

    // Synchronize to ensure all data is loaded into shared memory
    __syncthreads();

    // Apply the convolution using the shared memory
    if (gi < n) {
        float result = 0.0;
        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
            result += shared_mem[li + j] * kernel[KERNEL_RADIUS + j];
        }
        output[gi] = result;
    }
}


int main() {
    int size = N * sizeof(float);  // Size in bytes for input and output vectors

    // Host memory allocation
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    // Initialize input vector with input data
    // In this example: a sine wave + noise
    for (int i = 0; i < N; i++) {
        h_input[i] = 2.*sinf(i * 0.1) + (float)rand()/RAND_MAX;  
        // If willing to plot all inputs
        // printf("%f,",h_input[i]);
    }
    
    // Define a 1D smoothing kernel (e.g., a simple averaging kernel or Gaussian-like kernel)
    int kernel_size = 2 * KERNEL_RADIUS + 1;
    float h_kernel[kernel_size] = {0.004, 0.054, 0.242, 0.399, 0.242, 0.054, 0.004};  // Example: Gaussian-like kernel

    // Device memory allocation
    float *d_input, *d_output, *d_kernel;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMalloc((void**)&d_kernel, kernel_size * sizeof(float));

    // Copy input data and kernel from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int threadsPerBlock = THREADS_PER_BLOCK;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the 1D convolution kernel
    // convolve1D<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_kernel, N);
    convolve1D_sharedMemory<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_kernel, N);

    // Copy the result back from device to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print a few input and output values for verification
    printf("\n");
    for (int i = 0; i < 10; i++) {
        printf("h_input[%d] = %f\n", i, h_input[i]);
    }
    printf("\n");
    for (int i = 0; i < 10; i++) {
        printf("h_output[%d] = %f\n", i, h_output[i]);
    }
    printf("\n");

    // // If willing to plot all outputs
    // for (int i = 0; i < N; i++) {
    //     printf("%f,",h_output[i]);
    // }
    // printf("\n");

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}
