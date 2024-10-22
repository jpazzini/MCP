#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <assert.h>

#define WIDTH 2048                          // Define the vector width
#define N_BLOCKS  256                       // Define the number of blocks
#define THREADS_PER_BLOCK WIDTH/N_BLOCKS/2  // Define the number of threads in a block

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

// CUDA kernel to perform parallel reduction with fewer thread divergences
__global__ void reduction_optimized(const int* V, int* R, const int width) {

    // ...

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
        printf("%d\t", V[i]);
    }
    printf("\n");
}

int main(int argc, char** argv) {

    // Seed the random number generator with the current time
    srand(time(NULL));  // Ensure that rand() produces different sequences each run

    // Local vector hosted in memory, each with N elements
    std::vector<int> V(WIDTH), R(N_BLOCKS, 0), F(1, 0); // Initialize result vector to zeros for simplicity
    std::generate(V.begin(), V.end(), random_number); // Fill vector 'V' with random numbers
    
    printf("Vector V\n");
    print_vector(V.data(), 10);

    int sum_of_elems = 0;    
    for (auto& n : V)
       sum_of_elems += n;
    printf("Sum = %d\n",sum_of_elems);

    // Allocate memory for the vectors on the device

    // Copy the input vectors to the device

    // Define the block and grid dimensions

    // Launch the CUDA kernel to apply the parallel reduction

    // Copy the result vector back to the host

    // Perform the host sum of the elements of the (now small) resulting vector
    // OR, as an alternative..
    // Perform another parallel reduction step to return a scalar

    // Printout and verify the result of the reduction
    
    // Free the memory on the host and the GPU

    return 0;
}
