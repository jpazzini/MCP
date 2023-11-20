#include <stdio.h>

__global__ void a_kernel(void){

    // Not doing much...

}

int main(int argc, char **argv){

    printf("Hello world\n");
    
    a_kernel<<<1,1>>>();
    
    return 0;

}
