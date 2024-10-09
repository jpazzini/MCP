// Include the standard input/output library
#include <stdio.h>

// Main function, entry point of the program
int main() {

  // Variable to store the number of CUDA-capable devices
  int nDevices;

  // Get the number of CUDA-capable devices
  cudaGetDeviceCount(&nDevices);
  
  // Print the number of devices found
  printf("Number of devices: %d\n", nDevices);
  
  // Loop through each device
  for (int i = 0; i < nDevices; i++) {
    // Structure to hold properties of the device
    cudaDeviceProp prop;
    
    // Get the properties of the device
    cudaGetDeviceProperties(&prop, i);

    // Print the device properties
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  n SMPs: %d\n", prop.multiProcessorCount);
    printf("  n SPs: %d\n", prop.multiProcessorCount*128);
    printf("  Clock rate (MHz): %.1f\n", prop.clockRate/1024.);
    printf("  L2 Cache Size (KB): %.1f\n", prop.l2CacheSize*1e-3);
    printf("  Memory Clock Rate (MHz): %d\n",
           prop.memoryClockRate/1024);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Total global memory (GB) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
    printf("  Shared memory per block (KB) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
    printf("  minor-major: %d-%d\n", prop.minor, prop.major);
    printf("  Warp-size: %d\n", prop.warpSize);
    printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
    printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
  }
}
