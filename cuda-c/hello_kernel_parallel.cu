// Include the standard input/output library
#include <stdio.h>

// Define a CUDA kernel
__global__ void a_kernel(void){

    // Print a message from the GPU
    printf("Hello world from the GPU\n");
}

// Main function, entry point of the program
int main(int argc, char **argv){

    // Print "Hello world" to the console from the CPU
    printf("Hello world\n");
    
    // Launch the CUDA kernel with 1 block and 4 threads (asynchronous execution)
    a_kernel<<<1,4>>>();

    // Print a message from the CPU after kernel launch
    printf("Back on the CPU\n");
    
    // Destroy and clean up all resources of the current process on the device
    cudaDeviceSynchronize();
    // cudaDeviceReset();
    
    // Return 0 to indicate successful execution
    return 0;
}
