#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>

__global__ void sum_kernel(const int *x,const int *y, int *res, int max_index){
    // All operands are passed by reference

    // The operation is to be executed on the device,
    // so the variables x,y,res must point to the memory of the GPU

    // The index refers to the innermost thread execution unit
    // thus the thread inside a block
    auto index = threadIdx.x + blockIdx.x * blockDim.x;

    // With arbitrary-sized vectors, we must protect the kernel 
    // from accessing a piece of memory of the device that has not been allocated
    if (index < max_index){
    	res[index] = x[index] + y[index];
    }
    printf("Block - Thread number %d-%d\n",blockIdx.x,threadIdx.x);
}

int random_number(){
    return (std::rand()%100);
}

int main(int argc, char **argv){

    srand(time(NULL));

    int N = 39;                         // Length of the vectors
    
    std::vector<int> a(N), b(N), c(N);  // Local variables, hosted in memory
    std::generate(a.begin(), a.end(), random_number);
    std::generate(b.begin(), b.end(), random_number);
    
    int *dev_a, *dev_b, *dev_c;         // Copies of the variables hosted on the device

    int size = N * sizeof(int);         // Size of the memory associated to each entry

    // Carve the memory on the device by allocating space 
    // for the copies hosted on the device
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, size);

    for (int i = 0; i < N; i++){
        printf("[el. %d] %d + %d = %d (on CPU) \n",i,a[i],b[i],a[i]+b[i]);
    }
    
    // Copy input variables to the device
    cudaMemcpy(dev_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), size, cudaMemcpyHostToDevice);

    // Launch the sum kernel on the GPU
    //  - 40 parallel executions of the kernel  
    //  --> 5 blocks of 8 threads each

    // We must be careful not to execute a kernel on a non-existing piece of data

    int N_threadsPerBlock = 10;
    sum_kernel<<<(N + N_threadsPerBlock - 1)/N_threadsPerBlock,N_threadsPerBlock>>>(dev_a, dev_b, dev_c, N);
    
    // Copy the result from the device back on the local variables
    cudaMemcpy(c.data(), dev_c, size, cudaMemcpyDeviceToHost);    
   
    for (int i = 0; i < N; i++){
        printf("[el. %d] %d + %d = %d (on GPU) \n",i,a[i],b[i],c[i]);
    }
   
    // Cleanup by freeing all used memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;

}
