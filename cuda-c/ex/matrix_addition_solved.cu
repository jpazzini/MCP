#include <stdio.h>

#define ROWS 4000  // Number of rows in the matrices
#define COLS 6000  // Number of columns in the matrices

__global__ void matrixAdd(float* A, float* B, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // All cuda indexes are stored in registers
    int col = blockIdx.x * blockDim.x + threadIdx.x; // (same for `row` and `col`)

    if (row < rows && col < cols) {
        int idx = row * cols + col; // `idx` is stored in registers
        C[idx] = A[idx] + B[idx]; // 1 FP operation, 3 (2 read + 1 write) memory accesses
    }
}

int main() {
    // Size in bytes for the ROWS x COLS matrix
    int size = ROWS * COLS * sizeof(float);  

    // Host memory allocation
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize matrices A and B
    for (int i = 0; i < ROWS * COLS; i++) {
        h_A[i] = 1.0 + (float)rand()/RAND_MAX;
        h_B[i] = 2.0 + (float)rand()/RAND_MAX;
    }

    // Create the cudaEvent timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);  // 16x16 threads per block
    dim3 numBlocks((COLS + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (ROWS + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Assign the start cudaEvent timer
    cudaEventRecord(start);

    // Launch the kernel
    matrixAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, ROWS, COLS);

    // Assign the stop cudaEvent timer and synchronize
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy the result matrix C from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print part of the result matrix C for verification
    for (int i = 0; i < 10; ++i){
        printf("C[%d] (cpu): %2.3f\n", i, h_A[i]+h_B[i]);
        printf("C[%d] (gpu): %2.3f\n\n", i, h_C[i]);
    }

    // Print the time taken (in ms) between two events
    float elapsed_gpu;
    cudaEventElapsedTime(&elapsed_gpu, start, stop);
    printf("Elapsed time (gpu): %.2f ms\n", elapsed_gpu);

    // Destroy cudaEvents
    cudaEventDestroy(start);
    cudaEventDestroy(start);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
