#include <stdio.h>
#include <vector>     // For std::vector
#include <algorithm>  // For std::generate
#include <stdlib.h>   // For srand and rand
#include <time.h>     // For time

#define N 1021                  // Define the size of the vector
#define THREADS_PER_BLOCK 128   // Define the number of threads in a block

// Define a CUDA kernel that adds two vectors element-wise
__global__ void sum_kernel(const int *x, const int *y, int *res, int max_index){
    // Calculate the thread ID of the overall grid
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;

    printf("Block - Thread number %d-%d\n",blockIdx.x,threadIdx.x);

    // Each thread computes one element of the result vector
    // Protect the thread against accessing non-existing memory locations
    if (idx < max_index){
        res[idx] = x[idx] + y[idx];
    }
}

// Function to generate a random number between 0 and 99
int random_number() {
    return (std::rand() % 100);
}

int main(int argc, char **argv){

    // Seed the random number generator with the current time
    srand(time(NULL));  // Ensure that rand() produces different sequences each run

    // Local vectors hosted in memory, each with N elements
    std::vector<int> a(N), b(N), c(N);

    // Initialize vectors 'a' and 'b' with random numbers
    std::generate(a.begin(), a.end(), random_number);  // Fill vector 'a' with random numbers
    std::generate(b.begin(), b.end(), random_number);  // Fill vector 'b' with random numbers
    
    // Pointers to device (GPU) memory for the vectors
    int *dev_a, *dev_b, *dev_c;         

    // Determine the size of the memory required for each vector
    int size = N * sizeof(int);             

    // Allocate space on the GPU for the copies of the vectors
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, size);

    // Print the result of vector addition on the CPU
    printf("CPU result:\n");
    for (int i = 0; i < N; i++) {
        printf("[el. %d] %d + %d = %d (on CPU) \n",i,a[i],b[i],a[i]+b[i]);
    }
    
    // Copy the input vectors from the CPU to the GPU
    cudaMemcpy(dev_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), size, cudaMemcpyHostToDevice);

    // Compute the number of blocks and threads per block
    int N_b   = ceil(float(N)/THREADS_PER_BLOCK);
    int N_tpb = THREADS_PER_BLOCK;

    // Launch the sum kernel on the GPU with multiple blocks and threads
    // and protect against accessing non-existing data locations
    sum_kernel<<<N_b, N_tpb>>>(dev_a, dev_b, dev_c, N);
    
    // Copy the result vector from the GPU back to the CPU
    cudaMemcpy(c.data(), dev_c, size, cudaMemcpyDeviceToHost);    

    // Print the result of the vector addition performed on the GPU
    printf("GPU result:\n");
    for (int i = 0; i < N; i++) {
        printf("[el. %d] %d + %d = %d (on GPU) \n",i,a[i],b[i],c[i]);
    }
   
    // Cleanup by freeing the allocated GPU memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);        

    // Return 0 to indicate successful execution
    return 0;
}
