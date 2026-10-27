#include <stdio.h>
#include <stdlib.h>

#define KERNEL_SIZE 3                   // Size of the square kernel (KERNEL_SIZE x KERNEL_SIZE)
#define THREADS_PER_BLOCK_X 16          // Number of threads per block in X
#define THREADS_PER_BLOCK_Y 16          // Number of threads per block in Y
#define HALF_KERNEL (KERNEL_SIZE / 2)   // Half size of the kernel (integer division, i.e. 3/2 = 1)

// CUDA Kernel to apply a naive 2D kernel to the image
__global__ void convolve2d(int *d_input, int *d_output, int width, int height, int max_val, const float *d_kernel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        float sum = 0.0;
        
        // Iterate over the kernel
        for (int i = -HALF_KERNEL; i <= HALF_KERNEL; i++) {
            for (int j = -HALF_KERNEL; j <= HALF_KERNEL; j++) {
                int x = min(max(idx + i, 0), width - 1);
                int y = min(max(idy + j, 0), height - 1);
                
                sum += d_input[y * width + x] * d_kernel[(i + HALF_KERNEL) * KERNEL_SIZE + (j + HALF_KERNEL)];
            }
        }

        if (sum < 0){
            sum = 0.;
        } else if (sum > max_val){
            sum = (float)max_val;
        } 

        d_output[idy * width + idx] = (int)sum;
    }
}

// CUDA Kernel to apply a 2D kernel using shared memory (stencil pattern)
__global__ void convolve2d_sharedMemory(int *d_input, int *d_output, int width, int height, int max_val, const float *d_kernel) {

    // Shared memory
    __shared__ int shared_mem[THREADS_PER_BLOCK_X + 2 * HALF_KERNEL][THREADS_PER_BLOCK_Y + 2 * HALF_KERNEL];

    // Calculate global and local (within the block) thread indices
    int tx = threadIdx.x;                       // Local x thread index
    int ty = threadIdx.y;                       // Local y thread index
    int grow = blockIdx.y * blockDim.y + ty;    // Global y thread index, i.e. input 2d data row
    int gcol = blockIdx.x * blockDim.x + tx;    // Global x thread index, i.e. input 2d data column
    int lrow = ty + HALF_KERNEL;                // Local shared 2d data row
    int lcol = tx + HALF_KERNEL;                // Local shared 2d data column

    // Load the main tile plus halo regions into shared memory
    // Each thread loads its own pixel and may help with halo pixels
    // The logic for the load of the halos can be streamlined
    // Here's reported a more plain version for readability

    // Load the interior pixels into shared memory
    if (gcol < width && grow < height) {
        shared_mem[lrow][lcol] = d_input[grow * width + gcol];
    }
    
    // Left, Right, Top, Bottom
    if (tx < HALF_KERNEL && gcol >= HALF_KERNEL) {  // Left halo 
        shared_mem[lrow][tx] = d_input[grow * width + (gcol - HALF_KERNEL)];
    }
    if (tx >= blockDim.x - HALF_KERNEL && gcol + HALF_KERNEL < width) {  // Right halo
        shared_mem[lrow][lcol + HALF_KERNEL] = d_input[grow * width + (gcol + HALF_KERNEL)];
    }
    if (ty < HALF_KERNEL && grow >= HALF_KERNEL) {  // Top halo
        shared_mem[ty][lcol] = d_input[(grow - HALF_KERNEL) * width + gcol];
    }
    if (ty >= blockDim.y - HALF_KERNEL && grow + HALF_KERNEL < height) {  // Bottom halo
        shared_mem[lrow + HALF_KERNEL][lcol] = d_input[(grow + HALF_KERNEL) * width + gcol];
    }

    // Load corner halos
    if (tx < HALF_KERNEL && ty < HALF_KERNEL && gcol >= HALF_KERNEL && grow >= HALF_KERNEL) {  // Top-left corner
        shared_mem[ty][tx] = d_input[(grow - HALF_KERNEL) * width + (gcol - HALF_KERNEL)];
    }
    if (tx >= blockDim.x - HALF_KERNEL && ty < HALF_KERNEL && gcol + HALF_KERNEL < width && grow >= HALF_KERNEL) {  // Top-right corner
        shared_mem[ty][lcol + HALF_KERNEL] = d_input[(grow - HALF_KERNEL) * width + (gcol + HALF_KERNEL)];
    }
    if (tx < HALF_KERNEL && ty >= blockDim.y - HALF_KERNEL && gcol >= HALF_KERNEL && grow + HALF_KERNEL < height) {  // Bottom-left corner
        shared_mem[lrow + HALF_KERNEL][tx] = d_input[(grow + HALF_KERNEL) * width + (gcol - HALF_KERNEL)];
    }
    if (tx >= blockDim.x - HALF_KERNEL && ty >= blockDim.y - HALF_KERNEL && gcol + HALF_KERNEL < width && grow + HALF_KERNEL < height) {  // Bottom-right corner
        shared_mem[lrow + HALF_KERNEL][lcol + HALF_KERNEL] = d_input[(grow + HALF_KERNEL) * width + (gcol + HALF_KERNEL)];
    }

    // Synchronize to ensure all threads have loaded data into shared memory
    __syncthreads();

    // Apply the 2D convolution (kernel)
    if (gcol < width && grow < height) {
        float sum = 0.0f;

        // Iterate over the kernel
        for (int i = -HALF_KERNEL; i <= HALF_KERNEL; i++) {
            for (int j = -HALF_KERNEL; j <= HALF_KERNEL; j++) {
                sum += shared_mem[lrow + i][lcol + j] * d_kernel[(i + HALF_KERNEL) * KERNEL_SIZE + (j + HALF_KERNEL)];
            }
        }

        if (sum < 0){
            sum = 0.;
        } else if (sum > max_val){
            sum = (float)max_val;
        } 

        // Store the result in the output image
        d_output[grow * width + gcol] = (int)sum;
    }
}

// Function to read the PGM file (P2 format)
int *read_pgm(const char *filename, int *width, int *height, int *max_val) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Could not open file.\n");
        return NULL;
    }

    // Read the PGM header (P2 format, width, height, max_val)
    char format[3];
    fscanf(file, "%s", format);
    if (format[0] != 'P' || format[1] != '2') {
        printf("Error: Unsupported PGM format.\n");
        fclose(file);
        return NULL;
    }

    // Read image dimensions and maximum gray value
    fscanf(file, "%d %d", width, height);
    fscanf(file, "%d", max_val);

    int total_pixels = (*width) * (*height);

    // Allocate memory to store grayscale pixel data
    int *image = (int *)malloc(total_pixels * sizeof(int));
    if (image == NULL) {
        printf("Error: Could not allocate memory.\n");
        fclose(file);
        return NULL;
    }

    // Read pixel data
    for (int i = 0; i < total_pixels; i++) {
        fscanf(file, "%d", &image[i]);
    }

    fclose(file);  // Close the file
    return image;  // Return the pixel data array
}

// Function to write a PGM file
void write_pgm(const char *filename, int *image, int width, int height, int max_val) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error: Could not open file for writing.\n");
        return;
    }

    // Write the PGM header
    fprintf(file, "P2\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "%d\n", max_val);

    // Write the pixel values
    for (int i = 0; i < width * height; i++) {
        fprintf(file, "%d ", image[i]);
        if ((i + 1) % width == 0) {
            fprintf(file, "\n");
        }
    }

    fclose(file);  // Close the file
}

int main() {
    int width, height, max_val;

    // Read the PGM image
    int *host_input = read_pgm("ny_gray.pgm", &width, &height, &max_val);
    if (host_input == NULL) {
        return 1;  // Error reading the file
    }

    // Allocate memory for the output image on the host
    int *host_output = (int *)malloc(width * height * sizeof(int));
    if (host_output == NULL) {
        printf("Error: Could not allocate memory for output image.\n");
        free(host_input);
        return 1;
    }

    // Define the 3x3 kernel
    float kernel[KERNEL_SIZE * KERNEL_SIZE] = {
         0., -1.,  0.,
        -1.,  4,  -1.,
         0., -1.,  0.,
    };

    // Allocate memory for the image and kernel on the GPU
    int *device_input, *device_output;
    float *device_kernel;

    cudaMalloc((void **)&device_input, width * height * sizeof(int));
    cudaMalloc((void **)&device_output, width * height * sizeof(int));
    cudaMalloc((void **)&device_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    // Copy the input image and the kernel to the GPU
    cudaMemcpy(device_input, host_input, width * height * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_kernel, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Define the block and grid dimensions
    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 numBlocks((width + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X,
                   (height + THREADS_PER_BLOCK_Y - 1) / THREADS_PER_BLOCK_Y);

    // Launch the CUDA kernel to apply the convolution
    convolve2d_sharedMemory<<<numBlocks, threadsPerBlock>>>(device_input, device_output, width, height, max_val, device_kernel);

    // Copy the output image back to the host
    cudaMemcpy(host_output, device_output, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    // Write the image to a new PGM file
    write_pgm("output.pgm", host_output, width, height, max_val);

    // Free the memory on the host and the GPU
    free(host_input);
    free(host_output);
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_kernel);

    return 0;
}
