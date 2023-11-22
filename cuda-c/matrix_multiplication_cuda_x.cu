#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>

// CUDA kernel to perform matrix multiplication
__global__ void matrixMultiplication(const int* M, const int* N, int* P, const int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < width * width) {
        int row = idx / width;
        int col = idx % width;

        int p_value = 0;
        for (int k = 0; k < width; ++k) {
            int m_value = M[row * width + k];
            int n_value = N[k * width + col];
            p_value += m_value * n_value;
        }
        P[idx] = p_value;
    }
}

int random_number() {
    return (std::rand() % 100);
}

void print_matrix(const int* M, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d", M[i * n + j]);
            if (j == n) continue;
            printf("\t");
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    srand(time(NULL));

    const int n = 1024;  // Width of the matrices

    std::vector<int> A(n * n), B(n * n), C(n * n);  // Local variables, hosted in memory
    std::generate(A.begin(), A.end(), random_number);
    std::generate(B.begin(), B.end(), random_number);

    printf("Matrix A\n");
    //print_matrix(A.data(), n);

    printf("Matrix B\n");
    //print_matrix(B.data(), n);

    // Device matrices
    int* d_A;
    int* d_B;
    int* d_C;
    size_t matrixSize = n * n * sizeof(int);
    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_B, matrixSize);
    cudaMalloc((void**)&d_C, matrixSize);

    // Copy host matrices to device
    cudaMemcpy(d_A, A.data(), matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), matrixSize, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    int blockSize = 256;  // Adjust block size based on device capabilities
    int gridSize = (n * n + blockSize - 1) / blockSize;

    // Launch CUDA kernel
    matrixMultiplication<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

    // Copy result matrix from device to host
    cudaMemcpy(C.data(), d_C, matrixSize, cudaMemcpyDeviceToHost);

    printf("Matrix C\n");
    //print_matrix(C.data(), n);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

