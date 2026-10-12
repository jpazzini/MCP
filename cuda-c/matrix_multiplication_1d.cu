#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <assert.h>

#define WIDTH 2048              // Define the matrix width number
#define THREADS_PER_BLOCK 256   // Define the number of threads in a block

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes one element of the result matrix
    if (idx < width * width) {
        int row = idx / width;
        int col = idx % width;

        float sum = 0.;
        // Accessing all elements of a row of M and a column of N
        for (int k = 0; k < width; ++k) {
            sum += M[row * width + k] * N[k * width + col];
        }
        P[idx] = sum;
    }
}

// Function to perform matrix multiplication on the CPU (for verification)
void matrixMultiplicationCPU(const float* M, const float* N, float* P, const int width) {

    printf("\nComputing matrix multiplication on host... ");
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            for (int k = 0; k < width; ++k) {
                P[row * width + col] += M[row * width + k] * N[k * width + col];
            }
        }
        printf("\rComputing matrix multiplication on host... %3.f%%", 100.0 * (row + 1) / width);
    }
    printf("\n");
}

// Function to verify that two arrays are approximately equal
void allCloseTo(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        if (fabs(a[i] - b[i]) > tol) {
            printf("Mismatch at index %zu: %f vs %f\n", i, a[i], b[i]);
            assert(false);
        }
    }
    printf("All values are within the tolerance\n");
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
            printf("%8.2f", M[i * WIDTH + j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {

    // Seed the random number generator with the current time
    srand(time(NULL));  // Ensure that rand() produces different sequences each run

    // Local vectors hosted in memory, each with N elements
    // using a vector to host the matrix, in a row-wise allocation (row major)
    std::vector<float> M(WIDTH * WIDTH), N(WIDTH * WIDTH), P(WIDTH * WIDTH);
    std::vector<float> Pcpu(WIDTH * WIDTH, 0.);       // Fill vector 'Pcpu' with zeros
    std::generate(M.begin(), M.end(), random_number); // Fill vector 'M' with random numbers
    std::generate(N.begin(), N.end(), random_number); // Fill vector 'N' with random numbers

    printf("\nMatrix M\n");
    print_matrix(M.data(), 10, 10);

    printf("\nMatrix N\n");
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

    // Compute the number of blocks and threads per block
    // Blocks are 1-dimensional
    int N_b   = ceil(float(WIDTH * WIDTH)/THREADS_PER_BLOCK);
    int N_tpb = THREADS_PER_BLOCK;

    // Launch CUDA kernel
    matrixMultiplication<<<N_b, N_tpb>>>(d_M, d_N, d_P, WIDTH);

    // Copy the result vector from the GPU back to the CPU
    checkCuda(
        cudaMemcpy(P.data(), d_P, matrixSize, cudaMemcpyDeviceToHost)
    );

    printf("\nMatrix P\n");
    print_matrix(P.data(), 10, 10);

    /***
    // Run the same operation on CPU and test for closure
    // (comment these lines to save time)

    // Compute matrix multiplication on the CPU for verification
    matrixMultiplicationCPU(M.data(), N.data(), Pcpu.data(), WIDTH);

    // Verify the result (only if CPU computation was performed)
    allCloseTo(P, Pcpu, 1e-3);
    ***/

    // Cleanup by freeing the allocated GPU memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}

