#include <stdio.h>

// The number of elements is reduced to avoid running out of memory
#define M_ROWS 2000      // Number of rows in the matrix M
#define M_COLS 3000      // Number of cols in the matrix M
#define N_ROWS (M_COLS)  // Number of rows in the matrix N (matching the number of columns of matrix)
#define N_COLS 2500      // Number of cols in the matrix N

__global__ void matrixMultiplication(float* M, float* N, float* P, int rows_M, int cols_M, int rows_N, int cols_N) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes one element of the result matrix
    if (row < rows_M && col < cols_N) {
        float sum = 0.;
        // Accessing all ements of a row of M and a column of N
        for (int k = 0; k < cols_M; ++k) {
            sum += M[row * cols_M + k] * N[k * cols_N + col];
        }
        P[row * cols_N + col] = sum;
    }
}

// Function to printout the matrix (P)
void print_matrix(const float* M, int rows, int cols) {
    if (M_ROWS < rows)
        rows = M_ROWS;
    if (N_COLS < cols)
        cols = N_COLS;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.2f", M[i * N_COLS + j]);
        }
        printf("\n");
    }
}

int main() {
    // Size in bytes for the ROWS x COLS matrix
    int size_M = M_ROWS * M_COLS * sizeof(float);  
    int size_N = N_ROWS * N_COLS * sizeof(float);  
    int size_P = M_ROWS * N_COLS * sizeof(float);  

    // Host memory allocation
    float *h_M = (float*)malloc(size_M);
    float *h_N = (float*)malloc(size_N);
    float *h_P = (float*)malloc(size_P);

    // Initialize matrix M
    for (int i = 0; i < M_ROWS * M_COLS; i++) {
        h_M[i] = 1.0 + (float)rand()/RAND_MAX;
    }
    // Initialize matrix N
    for (int i = 0; i < N_ROWS * N_COLS; i++) {
        h_N[i] = 1.0 + (float)rand()/RAND_MAX;
    }

    // Device memory allocation
    float *d_M, *d_N, *d_P;
    cudaMalloc((void**)&d_M, size_M);
    cudaMalloc((void**)&d_N, size_N);
    cudaMalloc((void**)&d_P, size_P);

    // Copy matrices M and N from host to device
    cudaMemcpy(d_M, h_M, size_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);  
    dim3 gridSize((N_COLS + blockSize.x - 1) / blockSize.x, (M_ROWS + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    matrixMultiplication<<<gridSize, blockSize>>>(d_M, d_N, d_P, M_ROWS, M_COLS, N_ROWS, N_COLS);

    // Copy the result matrix P from device to host
    cudaMemcpy(h_P, d_P, size_P, cudaMemcpyDeviceToHost);

    // Print part of the result matrix P for verification
    print_matrix(h_P, 10, 10);

    // Free device memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    // Free host memory
    free(h_M);
    free(h_N);
    free(h_P);

    return 0;
}
