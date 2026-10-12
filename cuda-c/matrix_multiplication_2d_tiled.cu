#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <assert.h>

#define WIDTH 2048                      // Define the matrix width number
#define TILE_WIDTH 32                   // Define the tile width (same as thread block dimensions)
#define THREADS_PER_BLOCK_X TILE_WIDTH  // Define the number of threads in a block in x
#define THREADS_PER_BLOCK_Y TILE_WIDTH  // Define the number of threads in a block in y

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

// CUDA kernel to perform matrix multiplication
__global__ void matrixMultiplication(const float* M, const float* N, float* P, const int width) {
    // Declare shared matrices of size block*block (tiles)
    __shared__ float M_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N_tile[TILE_WIDTH][TILE_WIDTH];
    
    // For the sake of simplifying the notation, assign registers for thread_x,y
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate the row and column index of the current element
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    // Initialize the intermediate P value
    float sum = 0.;

    // Fill the shared memory
    //
    // Loop over the tiles of the input matrices
    // 
    // In the case `width` isn’t a multiple of TILE_WIDTH, the loop can be modified to:
    // for (int t = 0; t < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++t)
    for (int t = 0; t < width / TILE_WIDTH; ++t) {
        // Load (in collaboration with other threads) 
        // the tiles into shared memory
        if ( (row < width) && (t * TILE_WIDTH + tx < width) )
            M_tile[ty][tx] = M[row * width + t * TILE_WIDTH + tx];
        else 
            M_tile[ty][tx] = 0.;

        if ( (t * TILE_WIDTH + ty < width) && (col < width) )
            N_tile[ty][tx] = N[(t * TILE_WIDTH + ty) * width + col];
        else 
            N_tile[ty][tx] = 0.;

        // Synchronize (ensure the tile is loaded in shared memory)
        __syncthreads();
    
        // Perform the multiplication for this tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += M_tile[ty][k] * N_tile[k][tx];
        }

        // Ensure all threads are done computing before loading the next tile
        __syncthreads(); 
    }

    // Write the result back to the global memory
    if (row < width && col < width) {
        P[row * width + col] = sum;
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
    // using a vector to host the matrix, in a row-wise allocation
    std::vector<float> M(WIDTH * WIDTH), N(WIDTH * WIDTH), P(WIDTH * WIDTH);
    std::vector<float> Pcpu(WIDTH * WIDTH, 0.);       // Fill vector 'Pcpu' with zeros
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

