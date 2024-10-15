#include <stdio.h>
#include <stdlib.h>

// CUDA kernel to convert to grayscale
__global__ void rgb_to_grayscale(/* ... */) {    

    //...
}

// Function to read the PPM image into a 1D array
// (can be modified to read it into a 2D array or any other data types)
// 
int *read_pgm(const char *filename, int *width, int *height, int *max_val) {
    
    // Open the input file in read mode "r"
    FILE *file = fopen(filename, "r");
    
    // Check if file can be opened
    if (file == NULL) {
        printf("Could not open file.\n");
        return NULL;
    }

    // Read the PPM header, composed by 3 lines, e.g.:
    //
    // P3                           [magic number]
    // 1024 768                     [pixel_width pixel_height]
    // 65535                        [max color levels]
    //
    // More info here -- https://www.wikiwand.com/en/articles/Netpbm

    // Read the first line, and verify if it states `P3` 
    char format[3];
    fscanf(file, "%s", format);
    if (format[0] != 'P' || format[1] != '3') {
        printf("Not a valid PGM (ASCII P3) file.\n");
        fclose(file);
        return NULL;
    }

    // Read the width, height, and maximum grayscale value
    fscanf(file, "%d %d", width, height);
    fscanf(file, "%d", max_val);

    // Compute the total amount of pixels
    int total_pixels = (*width) * (*height);

    // Allocate host memory for the image data
    // 3 x the image size to allocate R G B values
    int *image = (int *)malloc(3 * total_pixels * sizeof(int));
    
    // Read pixel values into the array
    // R0 G0 B0 R1 G1 B1 ...
    for (int i = 0; i < total_pixels * 3; i++) {
        fscanf(file, "%d", &image[i]);
    }

    // Close the input file
    fclose(file);  

    // Return the pixel array
    return image;  
}

// Function to write the PGM image from a 1D array
void write_pgm(const char *filename, int *image, int width, int height, int max_val) {

    // Open the output file in write mode "w"
    FILE *file = fopen(filename, "w");

    // Check if file can be opened
    if (file == NULL) {
        printf("Could not open file for writing.\n");
        return;
    }

    // Write the PGM header
    fprintf(file, "P2\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "%d\n", max_val);

    // Write the pixel values
    for (int i = 0; i < width * height; i++) {
        fprintf(file, "%d ", image[i]);
        // Include a newline every "width" number of pixels
        if ((i + 1) % width == 0) {
            fprintf(file, "\n");
        }
    }

    // Close the output file
    fclose(file);  
}

int main() {
    int width, height, max_val;

    // Read the PGM image
    int *host_rgb = read_pgm("ny.ppm", &width, &height, &max_val);
    if (host_rgb == NULL) {
        return 1;  // Error reading the file
    }

    // Allocate memory for the grayscale image on the host

    // Allocate memory for the RGB image and Grayscale image on the GPU

    // Copy the RGB image data from the host to the device (GPU)
    
    // Define the block and grid dimensions
    
    // Launch the CUDA kernel to convert RGB to Grayscale

    // Copy the Grayscale image data back to the host
    
    // Write the grayscale image to a new PGM file
    
    // Free the memory on the host and the GPU

    return 0;
}
