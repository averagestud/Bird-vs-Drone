#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>              // Now clock_gettime and CLOCK_MONOTONIC are defined
#include "stb_image.h"
#include "model_weights.h"     // Contains conv1_weight, conv1_bias, etc.

// ----------------------------
// 1. Define Image and Layer Dimensions
// ----------------------------
#define IMG_HEIGHT 64
#define IMG_WIDTH  64
#define IMG_CHANNELS 3

// Block 1: Conv1 (3->32), kernel=3x3, BN1, then 2x2 maxpool
#define CONV1_OUT 32
#define POOL1_HEIGHT (IMG_HEIGHT / 2)   // 320
#define POOL1_WIDTH  (IMG_WIDTH / 2)     // 320

// Block 2: Conv2 (32->64), then pool -> 320x320 -> 160x160
#define CONV2_OUT 64
#define POOL2_HEIGHT (POOL1_HEIGHT / 2)   // 160
#define POOL2_WIDTH  (POOL1_WIDTH / 2)     // 160

// Block 3: Conv3 (64->128), then pool -> 160x160 -> 80x80
#define CONV3_OUT 128
#define POOL3_HEIGHT (POOL2_HEIGHT / 2)   // 80
#define POOL3_WIDTH  (POOL2_WIDTH / 2)     // 80

// Classifier: Global Average Pooling converts 80x80 feature map to a 128-element vector,
// then Fully Connected: 128 -> NUM_CLASSES (2)
#define NUM_CLASSES 2

// ----------------------------
// 2. Layer Function Definitions
// ----------------------------

// 2.1 Generic Convolution with Zero Padding (kernel size fixed to 3)
void conv2d_generic(int H, int W, int in_ch, int out_ch,
                    float in[H][W][in_ch],
                    float out[H][W][out_ch],
                    const float kernel[out_ch][in_ch][3][3],
                    const float bias[out_ch]) {
    int pad = 1;  // for kernel size 3
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int oc = 0; oc < out_ch; oc++) {
                float sum = 0.0f;
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            int ih = h + kh - pad;
                            int iw = w + kw - pad;
                            float pixel = 0.0f;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                pixel = in[ih][iw][ic];
                            sum += pixel * kernel[oc][ic][kh][kw];
                        }
                    }
                }
                out[h][w][oc] = sum + bias[oc];
            }
        }
    }
}

// 2.2 Batch Normalization: out = gamma * ((in - mean)/sqrt(var+eps)) + beta
void batch_norm_generic(int H, int W, int channels,
                        float in[H][W][channels],
                        float out[H][W][channels],
                        const float gamma[channels],
                        const float beta[channels],
                        const float running_mean[channels],
                        const float running_var[channels]) {
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < channels; c++) {
                out[h][w][c] = gamma[c] * ((in[h][w][c] - running_mean[c]) / sqrtf(running_var[c] + 1e-5f)) + beta[c];
            }
        }
    }
}

// 2.3 ReLU Activation (inplace)
void relu_generic(int H, int W, int channels, float data[H][W][channels]) {
    for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            for (int c = 0; c < channels; c++)
                if (data[h][w][c] < 0)
                    data[h][w][c] = 0;
}

// 2.4 Max Pooling (2x2, stride 2)
// in: [H][W][channels] -> out: [H/2][W/2][channels]
void maxpool2d_generic(int H, int W, int channels,
                       float in[H][W][channels],
                       float out[H/2][W/2][channels]) {
    int pool = 2;
    for (int h = 0; h < H/2; h++) {
        for (int w = 0; w < W/2; w++) {
            for (int c = 0; c < channels; c++) {
                float max_val = -1e9;
                for (int ph = 0; ph < pool; ph++) {
                    for (int pw = 0; pw < pool; pw++) {
                        int ih = h * pool + ph;
                        int iw = w * pool + pw;
                        float val = in[ih][iw][c];
                        if (val > max_val)
                            max_val = val;
                    }
                }
                out[h][w][c] = max_val;
            }
        }
    }
}

// 2.5 Global Average Pooling: in: [H][W][channels] -> out: [channels]
void global_avg_pool_generic(int H, int W, int channels,
                             float in[H][W][channels],
                             float out[channels]) {
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                sum += in[h][w][c];
            }
        }
        out[c] = sum / (H * W);
    }
}

// 2.6 Fully Connected Layer: in: [in_size], weights: [out_size][in_size], bias: [out_size]
void fully_connected_generic(int in_size, int out_size,
                             float in[in_size],
                             float out[out_size],
                             const float weights[out_size][in_size],
                             const float bias[out_size]) {
    for (int i = 0; i < out_size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < in_size; j++) {
            sum += in[j] * weights[i][j];
        }
        out[i] = sum + bias[i];
    }
}

// 2.7 Argmax: returns index of maximum element in an array of length len
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
// 3. Main Inference Pipeline (Using .bin file for image input)
// ----------------------------


int main(int argc, char** argv) {
    if(argc < 2) {
        printf("Usage: %s <image_bin_file>\n", argv[0]);
        return -1;
    }
    
    clock_t start, end;
    start = clock();

    char *img_filename = argv[1];
    
    // Read image from binary file.
    static float input_image[IMG_HEIGHT][IMG_WIDTH][IMG_CHANNELS];
    FILE *fp = fopen(img_filename, "rb");
    if(fp == NULL) {
        printf("Error: Could not open binary file %s\n", img_filename);
        return -1;
    }
    size_t num_elements = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS;
    size_t read = fread(input_image, sizeof(float), num_elements, fp);
    fclose(fp);
    if(read != num_elements) {
        printf("Error: Expected %zu elements, but read %zu\n", num_elements, read);
        return -1;
    }
    printf("Loaded binary image from %s\n", img_filename);
    
    // ----------------------------
    // Block 1: Conv1 -> BN1 -> ReLU -> MaxPool
    // ----------------------------
    static float block1_conv_out[IMG_HEIGHT][IMG_WIDTH][CONV1_OUT];
    conv2d_generic(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, CONV1_OUT,
                   input_image, block1_conv_out, conv1_weight, conv1_bias);
    
    static float block1_bn_out[IMG_HEIGHT][IMG_WIDTH][CONV1_OUT];
    batch_norm_generic(IMG_HEIGHT, IMG_WIDTH, CONV1_OUT,
                       block1_conv_out, block1_bn_out,
                       bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var);
    
    relu_generic(IMG_HEIGHT, IMG_WIDTH, CONV1_OUT, block1_bn_out);
    
    static float block1_pool_out[IMG_HEIGHT/2][IMG_WIDTH/2][CONV1_OUT];
    maxpool2d_generic(IMG_HEIGHT, IMG_WIDTH, CONV1_OUT,
                      block1_bn_out, block1_pool_out);
    
    // ----------------------------
    // Block 2: Conv2 -> BN2 -> ReLU -> MaxPool
    // ----------------------------
    static float block2_conv_out[IMG_HEIGHT/2][IMG_WIDTH/2][CONV2_OUT];
    conv2d_generic(IMG_HEIGHT/2, IMG_WIDTH/2, CONV1_OUT, CONV2_OUT,
                   block1_pool_out, block2_conv_out, conv2_weight, conv2_bias);
    
    static float block2_bn_out[IMG_HEIGHT/2][IMG_WIDTH/2][CONV2_OUT];
    batch_norm_generic(IMG_HEIGHT/2, IMG_WIDTH/2, CONV2_OUT,
                       block2_conv_out, block2_bn_out,
                       bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var);
    
    relu_generic(IMG_HEIGHT/2, IMG_WIDTH/2, CONV2_OUT, block2_bn_out);
    
    static float block2_pool_out[IMG_HEIGHT/4][IMG_WIDTH/4][CONV2_OUT];
    maxpool2d_generic(IMG_HEIGHT/2, IMG_WIDTH/2, CONV2_OUT,
                      block2_bn_out, block2_pool_out);
    
    // ----------------------------
    // Block 3: Conv3 -> BN3 -> ReLU -> MaxPool
    // ----------------------------
    static float block3_conv_out[IMG_HEIGHT/4][IMG_WIDTH/4][CONV3_OUT];
    conv2d_generic(IMG_HEIGHT/4, IMG_WIDTH/4, CONV2_OUT, CONV3_OUT,
                   block2_pool_out, block3_conv_out, conv3_weight, conv3_bias);
    
    static float block3_bn_out[IMG_HEIGHT/4][IMG_WIDTH/4][CONV3_OUT];
    batch_norm_generic(IMG_HEIGHT/4, IMG_WIDTH/4, CONV3_OUT,
                       block3_conv_out, block3_bn_out,
                       bn3_weight, bn3_bias, bn3_running_mean, bn3_running_var);
    
    relu_generic(IMG_HEIGHT/4, IMG_WIDTH/4, CONV3_OUT, block3_bn_out);
    
    static float block3_pool_out[IMG_HEIGHT/8][IMG_WIDTH/8][CONV3_OUT];
    maxpool2d_generic(IMG_HEIGHT/4, IMG_WIDTH/4, CONV3_OUT,
                      block3_bn_out, block3_pool_out);
    
    // ----------------------------
    // Global Average Pooling (GAP)
    // ----------------------------
    float gap_out[CONV3_OUT];  // Vector length = 128
    global_avg_pool_generic(IMG_HEIGHT/8, IMG_WIDTH/8, CONV3_OUT,
                            block3_pool_out, gap_out);
    
    // ----------------------------
    // Fully Connected Layer
    // ----------------------------
    float fc_out[NUM_CLASSES];
    fully_connected_generic(CONV3_OUT, NUM_CLASSES,
                            gap_out, fc_out, fc_weight, fc_bias);
    
    // ----------------------------
    // Argmax: Determine predicted class (0: Bird, 1: Drone)
    // ----------------------------
    int predicted = argmax_generic(NUM_CLASSES, fc_out);
    const char* result = (predicted == 0) ? "Bird" : "Drone";
    
    end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Inference time: %.6f seconds\n", time_taken);
    printf("Predicted Class: %s\n", result);
    
    return 0;
}
