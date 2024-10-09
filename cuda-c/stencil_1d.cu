#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <assert.h>

#define WIDTH 2048              // Define the vector length
#define RADIUS 3                // Define the radius of the stencil
#define THREADS_PER_BLOCK 256   // Define the number of threads in a block

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

// CUDA kernel to perform matrix multiplication
__global__ void stencil(const int* V, int* R, const int size, const int radius) {

    // Declare shared array of elements for the block
    // accounting for the two side radius
    __shared__ int tmp[THREADS_PER_BLOCK + 2 * RADIUS];

    // Calculate the global thread ID of the active kernel
    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate the indexes of the shared elements
    int s_idx = threadIdx.x + radius;

    // Fill the shared memory
    //
    // Copy an element from the global memory to the shared memory
    tmp[s_idx] = V[g_idx];
    // Check it the thread local index within its block 
    // (threadIdx.x) is less than the radius ("left border")
    if (threadIdx.x < radius) {
        // Load elements that are radius positions to the left
        // of the current element in the global memory vector V
        tmp[s_idx - radius] = V[g_idx - radius];
        // Load element that is blockDim.x positions to the right 
        // of the current element into shared memory 
        // (also fill up the "right border")
        tmp[s_idx + blockDim.x] = V[g_idx + blockDim.x];
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -radius ; offset <= radius ; offset++) {
        result += tmp[s_idx + offset];
    }
    
    // Store the result
    R[g_idx] = result;
}

// Function to generate a random number between 0 and 10
int random_number() {
    return (std::rand()%10);
}

// Function to printout the vector
void print_vector(const int* V, int len) {
    if (WIDTH < len) {
        len = WIDTH;
    }
    for (int i = 0; i < len; i++) {
        printf("%d\t", V[i]);
    }
    printf("\n");
}

int main(int argc, char** argv) {

    // Seed the random number generator with the current time
    srand(time(NULL));  // Ensure that rand() produces different sequences each run

    // Local vectors hosted in memory, each with N elements
    std::vector<int> V(WIDTH), R(WIDTH,0); // Initialize result vector to zeros for simplicity
    std::generate(V.begin(), V.end(), random_number); // Fill vector 'V' with random numbers

    printf("Vector V\n");
    print_vector(V.data(), 10);

    // Device vectors
    int* d_V;
    int* d_R;
    size_t vectorSize = WIDTH * sizeof(int);
    cudaMalloc((void**)&d_V, vectorSize);
    cudaMalloc((void**)&d_R, vectorSize);

    // Copy host vector to device
    cudaMemcpy(d_V, V.data(), vectorSize, cudaMemcpyHostToDevice);

    // Compute the number of blocks and threads per block
    // Blocks are 1-dimensional
    int N_b   = ceil(float(WIDTH)/THREADS_PER_BLOCK);
    int N_tpb = THREADS_PER_BLOCK;

    // Launch CUDA kernel
    stencil<<<N_b, N_tpb>>>(d_V, d_R, WIDTH, RADIUS);

    // Copy the result vector from the GPU back to the CPU
    checkCuda(
        cudaMemcpy(R.data(), d_R, vectorSize, cudaMemcpyDeviceToHost)
    );

    printf("Vector R\n");
    print_vector(R.data(), 10);

    // Cleanup by freeing the allocated GPU memory
    cudaFree(d_V);
    cudaFree(d_R);

    return 0;
}

