#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 1031  

// Kernel to transpose matrix M
__global__ void matrixTransposition(float *M, float *M_T, int width) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        int transposedIdx = col * width + row;  // Transposed index
        int originalIdx = row * width + col;    // Original index
        M_T[transposedIdx] = M[originalIdx];
    }
}

// Kernel to multiply matrices M_T (transposed M) and N
__global__ void matrixMultiplication(float *A_T, float *B, float *C, int width) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += A_T[k * width + row] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Function to printout the matrix
void print_matrix(const float* M, int rows, int cols) {
    if (WIDTH < rows)
        rows = WIDTH;
    if (WIDTH < cols)
        cols = WIDTH;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.2f", M[i * WIDTH + j]);
        }
        printf("\n");
    }
}

int main() {

    // Size in bytes 
    int size = WIDTH * WIDTH * sizeof(float);  

    // Host memory allocation
    float *h_M = (float*)malloc(size);
    float *h_N = (float*)malloc(size);
    float *h_P = (float*)malloc(size);

    // Initialize matrices M and N
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        h_M[i] = 1.0 + (float)rand()/RAND_MAX;
        h_N[i] = 2.0 + (float)rand()/RAND_MAX;
    }

    // Device memory allocation
    float *d_M, *d_N, *d_M_T, *d_P;
    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_M_T, size);
    cudaMalloc((void**)&d_P, size);

    // Copy matrices M and N from host to device
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (WIDTH + blockSize.y - 1) / blockSize.y);

    // Launch the matrixTransposition kernel (do NOT copy back the resulting M_T from Device to Host)
    matrixTransposition<<<gridSize, blockSize>>>(d_M, d_M_T, WIDTH);

    // Synch the device to wait for the previous kernel to be completed
    cudaDeviceSynchronize();

    // Launch the matrixMultiplication kernel
    matrixMultiplication<<<gridSize, blockSize>>>(d_M_T, d_N, d_P, WIDTH);

    // Copy the result matrix P from device to host
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    // Print part of the result matrix P for verification
    print_matrix(h_P, 10, 10);

    // Free device memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_M_T);
    cudaFree(d_P);

    // Free host memory
    free(h_M);
    free(h_N);
    free(h_P);

    return 0;
}
