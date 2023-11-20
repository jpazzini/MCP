#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>

__global__ void sum_kernel(const int *x,const int *y, int *res){
    // All operands are passed by reference

    // The operation is to be executed on the device,
    // so the variables x,y,res must point to the memory of the GPU

    // threadIdx is the entity that provides the index
    // of the thread to which the kernel is assigned within a block
    res[threadIdx.x] = x[threadIdx.x] + y[threadIdx.x];

    printf("Thread number %d\n",threadIdx.x);
}

int random_number(){
    return (std::rand()%100);
}

int main(int argc, char **argv){

    srand(time(NULL));

    int N = 4;                          // Length of the vectors
    
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
    //  - 4 parallel executions of the kernel  
    sum_kernel<<<1,4>>>(dev_a, dev_b, dev_c);
    
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
