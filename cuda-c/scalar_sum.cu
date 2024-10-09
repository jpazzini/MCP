#include <stdio.h>

// Define a CUDA kernel that adds two integers
__global__ void sum_kernel(int *x, int *y, int *res){
    // All operands are passed by reference
    //
    // The operation is executed on the device,
    // so the variables x, y, res must point to GPU memory
    
    *res = *x + *y;  // Perform addition on the GPU
}

int main(int argc, char **argv){

    // Local variables, hosted in the CPU memory
    int a, b, c;                        
    
    // Pointers to device (GPU) memory for the variables
    int *dev_a, *dev_b, *dev_c;         

    // Determine the size of the memory required for each integer
    int size = sizeof(int);             

    // Allocate space on the GPU for the copies of the variables (both inputs and output)
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, size);

    // Initialize the input variables on the CPU
    a = 2;
    b = 3;

    // Print the result of addition on the CPU
    printf("%d + %d = %d (on CPU) \n", a, b, a + b);
    
    // Copy the input variables from the CPU to the GPU
    cudaMemcpy(dev_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, &b, size, cudaMemcpyHostToDevice);

    // Launch the sum kernel on the GPU with 1 block and 1 thread (non-blocking operation)
    sum_kernel<<<1,1>>>(dev_a, dev_b, dev_c);
    
    // Copy the result from the GPU back to the CPU (blocking operation)
    cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost);    

    // Print the result of the addition performed on the GPU
    printf("%d + %d = %d (on GPU) \n", a, b, c);
   
    // Cleanup by freeing the allocated GPU memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);        

    return 0;

}
