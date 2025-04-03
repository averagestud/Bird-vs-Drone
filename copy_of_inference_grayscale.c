#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "model_weights_grayscale.h"  // Contains conv1_weight, bn1_bias, etc.

// ----------------------------
// 1. Define Image and Layer Dimensions (Grayscale)
// ----------------------------
#define IMG_HEIGHT 1024
#define IMG_WIDTH  1024
#define IMG_CHANNELS 1

// Block 1: Conv1 (1->32), kernel=3x3, BN1, then 2x2 maxpool
#define CONV1_OUT 32
#define POOL1_HEIGHT (IMG_HEIGHT / 2)
#define POOL1_WIDTH  (IMG_WIDTH / 2)

// Block 2: Conv2 (32->64)
#define CONV2_OUT 64
#define POOL2_HEIGHT (POOL1_HEIGHT / 2)
#define POOL2_WIDTH  (POOL1_WIDTH / 2)

// Block 3: Conv3 (64->128)
#define CONV3_OUT 128
#define POOL3_HEIGHT (POOL2_HEIGHT / 2)
#define POOL3_WIDTH  (POOL2_WIDTH / 2)

// Classifier: Global Average Pooling then FC: 128 -> NUM_CLASSES (2)
#define NUM_CLASSES 2

// ----------------------------
// 2. Layer Function Definitions (same as before)
// ----------------------------

void conv2d_generic(int H, int W, int in_ch, int out_ch,
                    float input[H][W][in_ch],
                    float output[H][W][out_ch],
                    const float kernel[out_ch][in_ch][3][3],
                    const float bias[out_ch]) {
    int pad = 1;
    // Assuming stride = 1 => output size equals H x W.
    #pragma omp parallel for collapse(2)
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int oc = 0; oc < out_ch; oc++) {
                float sum = bias[oc];
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            int ih = h + kh - pad;
                            int iw = w + kw - pad;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                sum += input[ih][iw][ic] * kernel[oc][ic][kh][kw];
                            }
                        }
                    }
                }
                output[h][w][oc] = sum;
            }
        }
    }
}

void batch_norm_generic(int H, int W, int channels,
                        float input[H][W][channels],
                        float output[H][W][channels],
                        const float gamma[channels],
                        const float beta[channels],
                        const float running_mean[channels],
                        const float running_var[channels]) {
    #pragma omp parallel for collapse(2)
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < channels; c++) {
                output[h][w][c] = gamma[c] * ((input[h][w][c] - running_mean[c]) /
                                   sqrtf(running_var[c] + 1e-5f)) + beta[c];
            }
        }
    }
}

void relu_generic(int H, int W, int channels, float data[H][W][channels]) {
    #pragma omp parallel for collapse(3)
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < channels; c++) {
                if (data[h][w][c] < 0)
                    data[h][w][c] = 0;
            }
        }
    }
}

void maxpool2d_generic(int H, int W, int channels,
                       float input[H][W][channels],
                       float output[(H)/2][(W)/2][channels]) {
    int pool = 2;
    #pragma omp parallel for collapse(2)
    for (int h = 0; h < H/2; h++) {
        for (int w = 0; w < W/2; w++) {
            for (int c = 0; c < channels; c++) {
                float max_val = -1e9;
                for (int ph = 0; ph < pool; ph++) {
                    for (int pw = 0; pw < pool; pw++) {
                        int ih = h * pool + ph;
                        int iw = w * pool + pw;
                        float val = input[ih][iw][c];
                        if (val > max_val)
                            max_val = val;
                    }
                }
                output[h][w][c] = max_val;
            }
        }
    }
}

void global_avg_pool_generic(int H, int W, int channels,
                             float input[H][W][channels],
                             float output[channels]) {
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        #pragma omp parallel for reduction(+:sum)
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                sum += input[h][w][c];
            }
        }
        output[c] = sum / (H * W);
    }
}

void fully_connected_generic(int in_size, int out_size,
                             float in[in_size],
                             float out[out_size],
                             const float weights[out_size][in_size],
                             const float bias[out_size]) {
    #pragma omp parallel for
    for (int o = 0; o < out_size; o++) {
        float sum = bias[o];
        for (int i = 0; i < in_size; i++) {
            sum += in[i] * weights[o][i];
        }
        out[o] = sum;
    }
}

int argmax_generic(int len, float in[len]) {
    int max_idx = 0;
    float max_val = in[0];
    for (int i = 1; i < len; i++) {
        if (in[i] > max_val) {
            max_val = in[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// ----------------------------
// 3. Main Inference Pipeline for Grayscale (Single Channel) 640x640 Image
// ----------------------------
int main(int argc, char** argv) {
    if(argc < 2) {
        printf("Usage: %s <image_bin_file>\n", argv[0]);
        return -1;
    }
    
    clock_t start, end;
    start = clock();
    
    char *img_filename = argv[1];
    
    // Allocate input image (expects binary file with IMG_HEIGHT * IMG_WIDTH * 1 floats)
    size_t num_elements = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS;
    float *input_image = (float *)malloc(sizeof(float) * num_elements);
    if(input_image == NULL) {
        printf("Error: Memory allocation failed for input_image\n");
        return -1;
    }
    
    FILE *fp = fopen(img_filename, "rb");
    if(fp == NULL) {
        printf("Error: Could not open binary file %s\n", img_filename);
        free(input_image);
        return -1;
    }
    size_t read = fread(input_image, sizeof(float), num_elements, fp);
    fclose(fp);
    if(read != num_elements) {
        printf("Error: Expected %zu elements, but read %zu\n", num_elements, read);
        free(input_image);
        return -1;
    }
    printf("Loaded %dx%d grayscale image from %s\n", IMG_HEIGHT, IMG_WIDTH, img_filename);
    
    // Allocate intermediate buffers dynamically
    float (*block1_conv_out)[IMG_WIDTH][CONV1_OUT] = malloc(sizeof(float) * IMG_HEIGHT * IMG_WIDTH * CONV1_OUT);
    float (*block1_bn_out)[IMG_WIDTH][CONV1_OUT]   = malloc(sizeof(float) * IMG_HEIGHT * IMG_WIDTH * CONV1_OUT);
    float (*block1_pool_out)[POOL1_WIDTH][CONV1_OUT]  = malloc(sizeof(float) * POOL1_HEIGHT * POOL1_WIDTH * CONV1_OUT);
    
    float (*block2_conv_out)[POOL1_WIDTH][CONV2_OUT]  = malloc(sizeof(float) * POOL1_HEIGHT * POOL1_WIDTH * CONV2_OUT);
    float (*block2_bn_out)[POOL1_WIDTH][CONV2_OUT]    = malloc(sizeof(float) * POOL1_HEIGHT * POOL1_WIDTH * CONV2_OUT);
    float (*block2_pool_out)[POOL2_WIDTH][CONV2_OUT]    = malloc(sizeof(float) * POOL2_HEIGHT * POOL2_WIDTH * CONV2_OUT);
    
    float (*block3_conv_out)[POOL2_WIDTH][CONV3_OUT]  = malloc(sizeof(float) * POOL2_HEIGHT * POOL2_WIDTH * CONV3_OUT);
    float (*block3_bn_out)[POOL2_WIDTH][CONV3_OUT]    = malloc(sizeof(float) * POOL2_HEIGHT * POOL2_WIDTH * CONV3_OUT);
    float (*block3_pool_out)[POOL3_WIDTH][CONV3_OUT]    = malloc(sizeof(float) * POOL3_HEIGHT * POOL3_WIDTH * CONV3_OUT);
    
    float *gap_out = malloc(sizeof(float) * CONV3_OUT);  // Global Average Pooling output vector (length = 128)
    float fc_out[NUM_CLASSES];
    
    if (!block1_conv_out || !block1_bn_out || !block1_pool_out ||
        !block2_conv_out || !block2_bn_out || !block2_pool_out ||
        !block3_conv_out || !block3_bn_out || !block3_pool_out || !gap_out) {
        printf("Error: Memory allocation failed for intermediate buffers\n");
        free(input_image);
        return -1;
    }
    
    // ----------------------------
    // Block 1: Conv1 -> BN1 -> ReLU -> MaxPool
    // ----------------------------
    conv2d_generic(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, CONV1_OUT,
                   (float (*)[IMG_WIDTH][IMG_CHANNELS])input_image,
                   block1_conv_out, conv1_weight, conv1_bias);
    
    batch_norm_generic(IMG_HEIGHT, IMG_WIDTH, CONV1_OUT,
                       block1_conv_out, block1_bn_out,
                       bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var);
    
    relu_generic(IMG_HEIGHT, IMG_WIDTH, CONV1_OUT, block1_bn_out);
    
    maxpool2d_generic(IMG_HEIGHT, IMG_WIDTH, CONV1_OUT,
                      block1_bn_out, block1_pool_out);
    
    // ----------------------------
    // Block 2: Conv2 -> BN2 -> ReLU -> MaxPool
    // ----------------------------
    conv2d_generic(POOL1_HEIGHT, POOL1_WIDTH, CONV1_OUT, CONV2_OUT,
                   block1_pool_out, block2_conv_out, conv2_weight, conv2_bias);
    
    batch_norm_generic(POOL1_HEIGHT, POOL1_WIDTH, CONV2_OUT,
                       block2_conv_out, block2_bn_out,
                       bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var);
    
    relu_generic(POOL1_HEIGHT, POOL1_WIDTH, CONV2_OUT, block2_bn_out);
    
    maxpool2d_generic(POOL1_HEIGHT, POOL1_WIDTH, CONV2_OUT,
                      block2_bn_out, block2_pool_out);
    
    // ----------------------------
    // Block 3: Conv3 -> BN3 -> ReLU -> MaxPool
    // ----------------------------
    conv2d_generic(POOL2_HEIGHT, POOL2_WIDTH, CONV2_OUT, CONV3_OUT,
                   block2_pool_out, block3_conv_out, conv3_weight, conv3_bias);
    
    batch_norm_generic(POOL2_HEIGHT, POOL2_WIDTH, CONV3_OUT,
                       block3_conv_out, block3_bn_out,
                       bn3_weight, bn3_bias, bn3_running_mean, bn3_running_var);
    
    relu_generic(POOL2_HEIGHT, POOL2_WIDTH, CONV3_OUT, block3_bn_out);
    
    maxpool2d_generic(POOL2_HEIGHT, POOL2_WIDTH, CONV3_OUT,
                      block3_bn_out, block3_pool_out);
    
    // ----------------------------
    // Global Average Pooling: reduce each (POOL3_HEIGHT x POOL3_WIDTH) feature map to one value per channel.
    // ----------------------------
    global_avg_pool_generic(POOL3_HEIGHT, POOL3_WIDTH, CONV3_OUT,
                            block3_pool_out, gap_out);
    
    // ----------------------------
    // Fully Connected Layer: 128 -> NUM_CLASSES (2)
    // ----------------------------
    fully_connected_generic(CONV3_OUT, NUM_CLASSES,
                            gap_out, fc_out, fc_weight, fc_bias);
    
    // ----------------------------
    // Argmax: Determine predicted class (e.g., 0: Bird, 1: Drone)
    // ----------------------------
    int predicted = argmax_generic(NUM_CLASSES, fc_out);
    const char* result = (predicted == 0) ? "Bird" : "Drone";
    
    end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Optimized Inference time: %.6f seconds\n", time_taken);
    printf("Predicted Class: %s\n", result);
    
    return 0;
}
