// Include the standard input/output library
#include <stdio.h>

// Define a simple CUDA kernel
__global__ void a_kernel(void){

    // Print a message from the GPU
    printf("Hello world from the GPU\n");
}

// Main function, entry point of the program
int main(int argc, char **argv){

    // Print "Hello world" to the console
    printf("Hello world\n");
    
    // Launch the CUDA kernel with a single block and a single thread
    a_kernel<<<1,1>>>();
    
    // Return 0 to indicate successful execution
    return 0;
}
