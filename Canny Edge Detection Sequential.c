#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "stb/stb_image.h"

#define M_PI 3.14159265358979323846

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Step 1: Gaussian smoothing
void gaussian_smooth(unsigned char *image, int width, int height, float sigma);
// Step 2: Gradient calculation
void calculate_gradients(unsigned char *image, int width, int height, float *magnitude, float *orientation);
// Step 3: Non-maximum suppression
void non_maximum_suppression(float *magnitude, float *orientation, unsigned char *edge_map, int width, int height);
// Step 4: Double thresholding
void double_thresholding(unsigned char *edge_map, int width, int height, float low_thresh, float high_thresh);
// Step 5: Edge tracking by hysteresis
void edge_tracking(unsigned char *edge_map, int width, int height, float low_thresh)



// Step 1: Gaussian smoothing
void gaussian_smooth(unsigned char *img, int width, int height, float sigma)
{
    int size = (int) (sigma * 6) + 1;
    int half_size = size / 2;
    float *kernel = (float *) malloc(size * sizeof(float));
    float sum = 0.0f;

    // Generate Gaussian kernel
    for (int i = 0; i < size; i++) {
        float x = i - half_size;
        kernel[i] = expf(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }

    // Normalize kernel
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }

    // Convolve image with Gaussian kernel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            for (int i = 0; i < size; i++) {
                int ix = x - half_size + i;
                if (ix < 0 || ix >= width) {
                    continue;
                }
                sum += kernel[i] * img[y * width + ix];
            }
            img[y * width + x] = (unsigned char) sum;
        }
    }

    free(kernel);
}

// Step 2: Gradient calculation
void calculate_gradients(unsigned char *img, int width, int height, float *magnitude, float *orientation)
{
    int kernel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int kernel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // Calculate gradient magnitude and orientation
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float dx = 0.0f, dy = 0.0f;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    dx += kernel_x[i][j] * img[(y + i - 1) * width + (x + j - 1)];
                    dy += kernel_y[i][j] * img[(y + i - 1) * width + (x + j - 1)];
                }
            }
            magnitude[y * width + x] = sqrtf(dx * dx + dy * dy);
            orientation[y * width + x] = atan2f(dy, dx);
        }
    }
}

// Step 3: Non-maximum suppression
void non_maximum_suppression(float *magnitude, float *orientation, unsigned char *edge_map, int width, int height)
{
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float mag = magnitude[y * width + x];
            float angle = orientation[y * width + x];
            float q = 255;
            float r = 255;

            // Check direction of edge
            if ((angle < -M_PI / 8) || (angle >= M_PI / 8 && angle <= 3 * M_PI / 8)) {
                q = magnitude[y * width + x + 1];
                r = magnitude[y * width + x - 1];
            } else if ((angle >= -3 * M_PI / 8 && angle < -M_PI / 8) || (angle >= 3 * M_PI / 8 && angle < M_PI / 8)) {
                q = magnitude[(y - 1) * width + x];
                r = magnitude[(y + 1) * width + x];
            } else if (angle >= M_PI / 8 && angle < 3 * M_PI / 8) {
                q = magnitude[y * width + x + 1];
                r = magnitude[y * width + x - 1];
            } else if (angle >= 3 * M_PI / 8 && angle < 5 * M_PI / 8) {
                q = magnitude[(y + 1) * width + x - 1];
                r = magnitude[(y - 1) * width + x + 1];
            }

            // Check if current pixel is a local maximum
            if (mag >= q && mag >= r) {
                edge_map[y * width + x] = (unsigned char) mag;
            } else {
                edge_map[y * width + x] = 0;
            }
        }
    }
}

// Step 4: Double thresholding
void double_thresholding(unsigned char *edge_map, int width, int height, float low_thresh, float high_thresh)
{
    // First pass: thresholding
    for (int i = 0; i < width * height; i++) {
        if (edge_map[i] >= high_thresh) {
            edge_map[i] = 255;
        } else if (edge_map[i] < low_thresh) {
            edge_map[i] = 0;
        }
    }

    // Second pass: hysteresis thresholding
    struct queue {
        int *elements;
        int capacity;
        int size;
        int front;
        int rear;
    };

    struct queue *edge_queue = (struct queue *)malloc(sizeof(struct queue));
    edge_queue->elements = (int *)malloc(width * height * sizeof(int));
    edge_queue->capacity = width * height;
    edge_queue->size = 0;
    edge_queue->front = 0;
    edge_queue->rear = -1;

    for (int i = 0; i < width * height; i++) {
        if (edge_map[i] == 255) {
            edge_queue->rear++;
            edge_queue->elements[edge_queue->rear] = i;
            edge_queue->size++;
        }
    }

    while (edge_queue->size > 0) {
        int i = edge_queue->elements[edge_queue->front];
        edge_queue->front++;
        edge_queue->size--;

        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                if (y == 0 && x == 0) {
                    continue;
                }
                int ix = i % width + x;
                int iy = i / width + y;
                if (ix < 0 || ix >= width || iy < 0 || iy >= height) {
                    continue;
                }
                if (edge_map[iy * width + ix] >= low_thresh && edge_map[iy * width + ix] < 255) {
                    edge_map[iy * width + ix] = 255;
                    edge_queue->rear++;
                    edge_queue->elements[edge_queue->rear] = iy * width + ix;
                    edge_queue->size++;
                }
            }
        }
    }

    free(edge_queue->elements);
    free(edge_queue);
}



// Step 5: Edge tracking by hysteresis
void edge_tracking(unsigned char *edge_map, int width, int height, float low_thresh)
{
    int weak = 25;
    int strong = 255;   //CHANGE BACK TO 75 OR 60 

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            if (edge_map[y * width + x] == weak) {
                // Check if any of the 8 neighboring pixels are strong
                if (edge_map[(y - 1) * width + (x - 1)] == strong ||
                    edge_map[(y - 1) * width + x] == strong ||
                    edge_map[(y - 1) * width + (x + 1)] == strong ||
                    edge_map[y * width + (x - 1)] == strong ||
                    edge_map[y * width + (x + 1)] == strong ||
                    edge_map[(y + 1) * width + (x - 1)] == strong ||
                    edge_map[(y + 1) * width + x] == strong ||
                    edge_map[(y + 1) * width + (x + 1)] == strong) {
                    edge_map[y * width + x] = strong;
                } else {
                    edge_map[y * width + x] = 0;
                }
            }
        }
    }
}

int main()
{
    FILE *fp;
    unsigned char *image_data;
    int width, height;
    float sigma, low_thresh, high_thresh;
    char filename[256];

    // Load input image from file
    printf("Enter input image filename: ");
    scanf("%s", filename);

    fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: could not open %s\n", filename);
        return -1;
    }

    fscanf(fp, "%*s %d %d %*s", &width, &height);

    image_data = (unsigned char *) malloc(width * height);
    fread(image_data, sizeof(unsigned char), width * height, fp);

    fclose(fp);

    // Set algorithm parameters
    printf("Enter Gaussian smoothing parameter sigma: ");
    scanf("%f", &sigma);

    printf("Enter double thresholding low threshold: ");
    scanf("%f", &low_thresh);

    printf("Enter double thresholding high threshold: ");
    scanf("%f", &high_thresh);

    // Apply Canny edge detection algorithm
    float *magnitude = (float *) calloc(width * height, sizeof(float));
    float *orientation = (float *) calloc(width * height, sizeof(float));
    unsigned char *edge_map = (unsigned char *) calloc(width * height, sizeof(unsigned char));

    gaussian_smooth(image_data, width, height, sigma);
    calculate_gradients(image_data, width, height, magnitude, orientation);
    non_maximum_suppression(magnitude, orientation, edge_map, width, height);
    double_thresholding(edge_map, width, height, low_thresh, high_thresh);
    edge_tracking(edge_map, width, height,sigma);
    

    // Save output image to file
    printf("Enter output image filename: ");
    scanf("%s", filename);

    fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error: could not open %s\n", filename);
        return -1;
    }


fprintf(fp, "P5\n%d %d\n255\n", width, height);
    fwrite(edge_map, sizeof(unsigned char), width * height, fp);

    fclose(fp);

    free(image_data);
    free(magnitude);
}
    
   

