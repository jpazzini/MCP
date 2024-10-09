#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <assert.h>

#define WIDTH 2048              // Define the matrix width number
#define THREADS_PER_BLOCK_X 32  // Define the number of threads in a block
#define THREADS_PER_BLOCK_Y 32  // Define the number of threads in a block

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

// CUDA kernel to perform matrix multiplication
__global__ void matrixMultiplication(const float* M, const float* N, float* P, const int width) {
    // Calculate the thread ID of the overall grid
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes one element of the result matrix
    if (row < width && col < width) {
        float sum = 0.;
        // Accessing all ements of a row of M and a column of N
        for (int k = 0; k < width; ++k) {
            sum += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = sum;
    }
}

// Function to generate a random number between 0 and 1
float random_number() {
    return (std::rand()*1./RAND_MAX);
}

// Function to printout the matrix
void print_matrix(const float* M, int rows, int cols) {
    if (WIDTH < rows) 
        rows = WIDTH;
    if (WIDTH < cols) 
        cols = WIDTH;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f", M[i * rows + j]);
            if (j < cols - 1) printf("\t");
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {

    // Seed the random number generator with the current time
    srand(time(NULL));  // Ensure that rand() produces different sequences each run

    // Local vectors hosted in memory, each with N elements
    // using a vector to host the matrix, in a row-wise allocation
    std::vector<float> M(WIDTH * WIDTH), N(WIDTH * WIDTH), P(WIDTH * WIDTH);
    std::generate(M.begin(), M.end(), random_number); // Fill vector 'M' with random numbers
    std::generate(N.begin(), N.end(), random_number); // Fill vector 'N' with random numbers

    printf("Matrix M\n");
    print_matrix(M.data(), 10, 10);

    printf("Matrix N\n");
    print_matrix(N.data(), 10, 10);

    // Device matrices
    float* d_M;
    float* d_N;
    float* d_P;
    size_t matrixSize = WIDTH * WIDTH * sizeof(float);
    cudaMalloc((void**)&d_M, matrixSize);
    cudaMalloc((void**)&d_N, matrixSize);
    cudaMalloc((void**)&d_P, matrixSize);

    // Copy host matrices to device
    cudaMemcpy(d_M, M.data(), matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N.data(), matrixSize, cudaMemcpyHostToDevice);

    // Compute the dimensions of blocks and grid
    // Blocks are now 2-dimensional
    dim3 blockSize(THREADS_PER_BLOCK_X,THREADS_PER_BLOCK_Y);
    dim3 gridSize(ceil(float(WIDTH)/blockSize.x),ceil(float(WIDTH)/blockSize.y));

    // Launch CUDA kernel
    matrixMultiplication<<<gridSize, blockSize>>>(d_M, d_N, d_P, WIDTH);

    // Copy the result vector from the GPU back to the CPU
    checkCuda(
        cudaMemcpy(P.data(), d_P, matrixSize, cudaMemcpyDeviceToHost)
    );

    printf("Matrix P\n");
    print_matrix(P.data(), 10, 10);

    // Cleanup by freeing the allocated GPU memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}

