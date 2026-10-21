#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <assert.h>

#define WIDTH 16384                         // Define the vector width
#define N_BLOCKS  32                        // Define the number of blocks
#define THREADS_PER_BLOCK WIDTH/N_BLOCKS/2  // Define the number of threads in a block

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

// CUDA kernel to perform parallel reduction
__global__ void reduction_naive(const int* V, int* R) {
    // Local index (within block) of thread
    int bdim = blockDim.x;
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // Global index of the data element 
    int start_idx = 2 * bdim * bx; // 2x as we are launching 1 thread every 2 items 

    // Shared memory to store partial sums
    // Limited to block, thus overall size 2x
    __shared__ int partialSum[2 * THREADS_PER_BLOCK];

    // Fill partial sum with two elements:
    // - the one corresponding to the thread
    // - the one 1 block size away from it
    partialSum[tx] = V[start_idx + tx];
    partialSum[tx + bdim] = V[start_idx + tx + bdim];

    // Loop over the shared memory doubling the stride
    // and sum values in place
    for (int stride = 1; stride <= bdim; stride *= 2) {

        // Ensure all elements of partial sums have been
        // generated before proceeding to the next step
        __syncthreads();

        // If the thread is active at this step, sum
        if (tx % stride == 0)
            partialSum[2 * tx] += partialSum[2 * tx + stride];
    }

    // Ensure all threads are done computing before 
    // offloading the result to global memory
    __syncthreads(); 
    // Write the result of this block to the output
    if (tx == 0) {
        R[blockIdx.x] = partialSum[0];
    }
}

// Function to generate a random number between 0 and 9
int random_number() {
    return (std::rand()%10);
}

// Function to printout the vector
void print_vector(const int* V, int len) {
    if (WIDTH < len) {
        len = WIDTH;
    }
    for (int i = 0; i < len; i++) {
        printf("%6d,", V[i]);
    }
    printf("\n");
}

int main(int argc, char** argv) {

    // Seed the random number generator with the current time
    srand(time(NULL));  // Ensure that rand() produces different sequences each run

    // Local vector hosted in memory, each with N elements
    std::vector<int> V(WIDTH), R(N_BLOCKS, 0); // Initialize result vector to zeros for simplicity
    std::generate(V.begin(), V.end(), random_number); // Fill vector 'V' with random numbers

    // Print the input vector
    printf("\nVector V (size %zu)\n",V.size());
    print_vector(V.data(), 10);

    // Perform the sum on the host
    int sum_of_elems = 0;
    for (auto& n : V)
       sum_of_elems += n;
    printf("Sum (cpu) = %d\n",sum_of_elems);

    // Device vectors
    int* d_V;
    int* d_R;
    size_t vectorSize = WIDTH * sizeof(int);
    size_t resultsSize = N_BLOCKS * sizeof(int);
    cudaMalloc((void**)&d_V, vectorSize);
    cudaMalloc((void**)&d_R, resultsSize);

    // Copy host vector to device
    cudaMemcpy(d_V, V.data(), vectorSize, cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    reduction_naive<<<N_BLOCKS, THREADS_PER_BLOCK>>>(d_V, d_R);

    // Copy the result vector from the GPU back to the CPU
    checkCuda(
        cudaMemcpy(R.data(), d_R, resultsSize, cudaMemcpyDeviceToHost)
    );

    // Print the resulting vector from the GPU-based reduction
    printf("\nVector R (size %zu)\n",R.size());
    print_vector(R.data(), 10);

    // Complete the final stage of reduction on the host
    // (alternatively, can be executed on GPU using 1 block)
    sum_of_elems = 0;
    for (auto& n : R)
       sum_of_elems += n;
    printf("Sum (gpu) = %d\n",sum_of_elems);


    // Cleanup by freeing the allocated GPU memory
    cudaFree(d_V);
    cudaFree(d_R);

    return 0;
}

