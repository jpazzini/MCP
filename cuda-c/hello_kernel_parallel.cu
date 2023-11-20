#include <stdio.h>

__global__ void a_kernel(void){

    printf("Hello world from the GPU\n");

}

int main(int argc, char **argv){

    printf("Hello world\n");
    
    // Asyncronous execution
    a_kernel<<<1,4>>>();

    printf("Back on the CPU\n");
    
    // Destroy and cleanup all resources of the current process on the device
    cudaDeviceReset();
    
    return 0;

}
