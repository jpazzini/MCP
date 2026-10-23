#include <stdio.h>
#include <stdlib.h>

#define KERNEL_SIZE 3  // Size of the square kernel (KERNEL_SIZE x KERNEL_SIZE)
#define THREADS_PER_BLOCK_X 16 // Number of threads per block in X
#define THREADS_PER_BLOCK_Y 16 // Number of threads per block in Y
#define HALF_KERNEL (KERNEL_SIZE / 2)  // Half size of the kernel

// CUDA Kernel to apply a naive 2D kernel to the image
__global__ void convolve2d(int *d_input, int *d_output, int width, int height, int max_val, const float *d_kernel) {

    // ...
}

// CUDA Kernel to apply a 2D kernel using shared memory (stencil pattern)
__global__ void convolve2d_sharedMemory(int *d_input, int *d_output, int width, int height, int max_val, const float *d_kernel) {

    // ...
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

    // Define the 3x3 kernel (e.g. ridge detection)
    float kernel[KERNEL_SIZE * KERNEL_SIZE] = {
         0., -1.,  0.,
        -1.,  4,  -1.,
         0., -1.,  0.,
    };

    // Allocate memory for the image and kernel on the GPU

    // Copy the input image and the kernel to the GPU

    // Define the block and grid dimensions

    // Launch the CUDA kernel to apply the convolution

    // Copy the output image back to the host

    // Write the image to a new PGM file

    // Free the memory on the host and the GPU

    return 0;
}
