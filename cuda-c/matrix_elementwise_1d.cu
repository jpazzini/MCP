#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>

#define ALPHA 0.25              // Define the scalar
#define ROWS 2160               // Define the matrix row number
#define COLS 4096               // Define the matrix column number
#define THREADS_PER_BLOCK 256   // Define the number of threads in a block

// CUDA kernel to perform elementwise multiplication
__global__ void matrix_elementwise(const float* M, float* P, const float alpha, const int rows, const int cols) {
    // Calculate the thread ID of the overall grid
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes one element of the result matrix
    if (idx < rows * cols) {
        P[idx] = alpha*M[idx];
    }
}

// Function to generate a random number between 0 and 99
float random_number() {
    return (std::rand()*100./RAND_MAX);
}

// Function to printout the matrix
void print_matrix(const float* M, int rows, int cols) {
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
    std::vector<float> M(ROWS * COLS), P(ROWS * COLS);
    std::generate(M.begin(), M.end(), random_number);  // Fill vector 'M' with random numbers

    printf("Matrix M\n");
    print_matrix(M.data(), 10, 10);

    // Device matrices
    float* d_M;
    float* d_P;
    size_t matrixSize = ROWS * COLS * sizeof(float);
    cudaMalloc((void**)&d_M, matrixSize);
    cudaMalloc((void**)&d_P, matrixSize);

    // Copy host matrix to device
    cudaMemcpy(d_M, M.data(), matrixSize, cudaMemcpyHostToDevice);

    // Compute the number of blocks and threads per block
    // Blocks are 1-dimensional
    int N_b   = ceil(float(ROWS)*COLS/THREADS_PER_BLOCK);
    int N_tpb = THREADS_PER_BLOCK;

    // Launch the CUDA kernel
    matrix_elementwise<<<N_b, N_tpb>>>(d_M, d_P, ALPHA, ROWS, COLS);

    // Copy the result vector from the GPU back to the CPU
    cudaMemcpy(P.data(), d_P, matrixSize, cudaMemcpyDeviceToHost);

    printf("Matrix P\n");
    print_matrix(P.data(), 10, 10);

    // Cleanup by freeing the allocated GPU memory
    cudaFree(d_M);
    cudaFree(d_P);

    return 0;
}

