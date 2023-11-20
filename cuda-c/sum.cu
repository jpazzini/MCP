#include <stdio.h>

__global__ void sum_kernel(int *x, int *y, int *res){
    // All operands are passed by reference

    // The operation is to be executed on the device,
    // so the variables x,y,res must point to the memory of the GPU
    
    *res = *x + *y;

}

int main(int argc, char **argv){

    int a, b, c;                        // Local variables, hosted in memory
    int *dev_a, *dev_b, *dev_c;         // Copies of the variables hosted on the device

    int size = sizeof(int);             // Size of the memory associated to each entry

    // Carve the memory on the device by allocating space 
    // for the copies hosted on the device
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, size);

    // Initialize the input variables    
    a = 2;
    b = 3;

    printf("%d + %d = %d (on CPU) \n",a,b,a+b);
    
    // Copy input variables to the device
    cudaMemcpy(dev_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, &b, size, cudaMemcpyHostToDevice);

    // Launch the sum kernel on the GPU
    sum_kernel<<<1,1>>>(dev_a, dev_b, dev_c);
    
    // Copy the result from the device back on the local variables
    cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost);    
   
    printf("%d + %d = %d (on GPU) \n",a,b,c);
   
    // Cleanup by freeing all used memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);        

    return 0;

}
