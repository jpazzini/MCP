#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <assert.h>

#define WIDTH 4096                          // Define the vector width
#define N_BLOCKS  8                         // Define the number of blocks
#define THREADS_PER_BLOCK WIDTH/N_BLOCKS/2  // Define the number of threads in a block

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

// CUDA kernel to perform parallel reduction with fewer thread divergences
__global__ void reduction_optimized(const int* V, int* R) {
    // Local index (within block) of thread
    int bdim = blockDim.x;
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // Global index of the data element 
    int start_idx = 2 * bdim * bx; // 2x as we are launching 1 thread every 2 items 

    // Shared memory to store partial sums
    __shared__ int partialSum[2 * THREADS_PER_BLOCK];

    // Load two elements into shared memory
    partialSum[tx] = V[start_idx + tx];
    partialSum[tx + bdim] = V[start_idx + tx + bdim];

    // Ensure the shared memory is fully populated
    __syncthreads();

    // Perform reduction with fewer thread divergences by adding elements that are farther apart in the beginning
    for (int stride = bdim; stride > 0; stride >>= 1) {
        // Only threads with index less than stride participate in this round
        if (tx < stride) {
            partialSum[tx] += partialSum[tx + stride];
        }
        // Synchronize threads within a block to ensure summation is complete
        __syncthreads();
    }

    // The first thread in the block writes the result of this block to the output array
    if (tx == 0) {
        R[blockIdx.x] = partialSum[0];
    }
}

// Function to generate a random number between 0 and 10
int random_number() {
    return (std::rand()%10);
}

// Function to print the vector
void print_vector(const int* V, int len) {
    if (WIDTH < len) {
        len = WIDTH;
    }
    for (int i = 0; i < len; i++) {
        printf("%5d ", V[i]);
    }
    printf("\n");
}

int main(int argc, char** argv) {

    // Seed the random number generator with the current time
    srand(time(NULL));  // Ensure that rand() produces different sequences each run

    // Local vector hosted in memory, each with N elements
    std::vector<int> V(WIDTH), R(N_BLOCKS, 0); // Initialize result vector to zeros for simplicity
    std::generate(V.begin(), V.end(), random_number); // Fill vector 'V' with random numbers
    int O = 0;
    
    printf("Input vector (truncated)\n");
    print_vector(V.data(), 12);

    int sum_of_elems = 0;    
    for (auto& n : V)
       sum_of_elems += n;
    printf("Sum (CPU) = %d\n",sum_of_elems);

    // Device vectors
    int* d_V;
    int* d_R;
    int* d_O;
    size_t vectorSize = WIDTH * sizeof(int);
    size_t resultsSize = N_BLOCKS * sizeof(int);
    cudaMalloc((void**)&d_V, vectorSize);
    cudaMalloc((void**)&d_R, resultsSize);
    cudaMalloc((void**)&d_O, sizeof(int));

    // Copy host vector to device
    cudaMemcpy(d_V, V.data(), vectorSize, cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    reduction_optimized<<<N_BLOCKS, THREADS_PER_BLOCK>>>(d_V, d_R);

    // Copy the result vector from the GPU back to the CPU
    checkCuda(
        cudaMemcpy(R.data(), d_R, resultsSize, cudaMemcpyDeviceToHost)
    );

    printf("Result vector (1 output element per block)\n");
    print_vector(R.data(), N_BLOCKS);

    // Perform final sum on CPU
    sum_of_elems = 0;    
    for (auto& n : R)
       sum_of_elems += n;
    printf("Sum (GPU+CPU) = %d\n",sum_of_elems);

    // Altenatively, launch the reduction kernel once again
    reduction_optimized<<<1, N_BLOCKS>>>(d_R, d_O, N_BLOCKS);

    // Copy the result vector from the GPU back to the CPU
    checkCuda(
        cudaMemcpy(&O, d_O, sizeof(int), cudaMemcpyDeviceToHost)
    );
    printf("Sum (GPU+GPU) = %d\n", O);

    // Cleanup by freeing the allocated GPU memory
    cudaFree(d_V);
    cudaFree(d_R);
    cudaFree(d_O);

    return 0;
}
