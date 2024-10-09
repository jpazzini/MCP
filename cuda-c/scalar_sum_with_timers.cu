#include <stdio.h>

// Define a CUDA kernel that adds two floats
__global__ void sum_kernel(float *x, float *y, float *res){
    // All operands are passed by reference
    //
    // The operation is executed on the device,
    // so the variables x, y, res must point to GPU memory
    
    *res = *x + *y;  // Perform addition on the GPU
}

int main(int argc, char **argv){

    // Local variables, hosted in the CPU memory
    float a, b, c;                        
    
    // Pointers to device (GPU) memory for the variables
    float *dev_a, *dev_b, *dev_c;         

    // Determine the size of the memory required for each float
    int size = sizeof(float);             

    // Create the cudaEvent timers
    cudaEvent_t start_alloc, start_copy, start_kernel, stop_kernel, stop_copy;
    cudaEventCreate(&start_alloc);
    cudaEventCreate(&start_copy);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&stop_copy);

    // Assign the start_allocation cudaEvent timer
    cudaEventRecord(start_alloc);

    // Allocate space on the GPU for the copies of the variables (both inputs and output)
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, size);

    // Initialize the input variables on the CPU
    a = 2.;
    b = 3.;

    // Print the result of addition on the CPU
    printf("%.2f + %.2f = %.2f (on CPU) \n", a, b, a + b);
    
    // Assign the start_copy cudaEvent timer
    cudaEventRecord(start_copy);

    // Copy the input variables from the CPU to the GPU
    cudaMemcpy(dev_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, &b, size, cudaMemcpyHostToDevice);

    // Assign the start_kernel cudaEvent timer
    cudaEventRecord(start_kernel);
    
    // Launch the sum kernel on the GPU with 1 block and 1 thread (non-blocking operation)
    sum_kernel<<<1,1>>>(dev_a, dev_b, dev_c);

    // Assign the stop_kernel cudaEvent timer and synchronize
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);

    // Copy the result from the GPU back to the CPU (blocking operation)
    cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost);    

    // Assign the stop_copy cudaEvent timer and synchronize
    cudaEventRecord(stop_copy);
    cudaEventSynchronize(stop_copy);

    // Print the result of the addition performed on the GPU
    printf("%.2f + %.2f = %.2f (on GPU) \n", a, b, c);
   
    // Print the time taken (in ms) between two events
    float elapsed_kernel, elapsed_copy, elapsed_alloc_to_copy;
    
    cudaEventElapsedTime(&elapsed_kernel,start_kernel, stop_kernel); // passing elapsed by reference
    cudaEventElapsedTime(&elapsed_copy,start_copy, stop_copy); 
    cudaEventElapsedTime(&elapsed_alloc_to_copy,start_alloc, stop_copy); 

    printf("Elapsed time (kernel):                 %.1f us\n", elapsed_kernel*1000);
    printf("Elapsed time (kernel+copy):            %.1f us\n", elapsed_copy*1000);
    printf("Elapsed time (kernel+copy+allocation): %.1f us\n", elapsed_alloc_to_copy*1000);

    // Destroy cudaEvents
    cudaEventDestroy(start_alloc);
    cudaEventDestroy(start_copy);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(stop_copy);

    // Cleanup by freeing the allocated GPU memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);        

    return 0;

}
