#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "model_weights_depthwise_classification.h"  // Auto-generated weights from Python

// ====================================
// 1. Define Image and Model Parameters
// ====================================
#define IMG_HEIGHT 1024
#define IMG_WIDTH  1024
#define IMG_CHANNELS 1   // Change to 1 for grayscale model

#define DWCONV_KERNEL 3
#define STRIDE 1
#define PADDING 1

#define BLOCK1_OUT 32
#define BLOCK2_OUT 64
#define BLOCK3_OUT 128
#define NUM_CLASSES 2

// ====================================
// 2. Depthwise + Pointwise Convolution
// ====================================

void depthwise_conv2d(int H, int W, int in_ch,
                      float input[H][W][in_ch],
                      float output[H][W][in_ch],
                      const float kernel[in_ch][1][DWCONV_KERNEL][DWCONV_KERNEL]) {
    int pad = PADDING;
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < in_ch; c++) {
                float sum = 0.0;
                for (int kh = 0; kh < DWCONV_KERNEL; kh++) {
                    for (int kw = 0; kw < DWCONV_KERNEL; kw++) {
                        int ih = h + kh - pad;
                        int iw = w + kw - pad;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            sum += input[ih][iw][c] * kernel[c][0][kh][kw];
                        }
                    }
                }
                output[h][w][c] = sum;
            }
        }
    }
}

void pointwise_conv2d(int H, int W, int in_ch, int out_ch,
                      float input[H][W][in_ch],
                      float output[H][W][out_ch],
                      const float kernel[out_ch][in_ch][1][1]) {
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int oc = 0; oc < out_ch; oc++) {
                float sum = 0.0;
                for (int ic = 0; ic < in_ch; ic++) {
                    sum += input[h][w][ic] * kernel[oc][ic][0][0];
                }
                output[h][w][oc] = sum;
            }
        }
    }
}

// ====================================
// 3. BatchNorm + ReLU
// ====================================

void batch_norm_relu(int H, int W, int channels,
                     float input[H][W][channels],
                     const float gamma[channels],
                     const float beta[channels],
                     const float mean[channels],
                     const float var[channels]) {
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < channels; c++) {
                float norm = (input[h][w][c] - mean[c]) / sqrtf(var[c] + 1e-5f);
                input[h][w][c] = fmaxf(0.0f, gamma[c] * norm + beta[c]);  // fused ReLU
            }
        }
    }
}

// ====================================
// 4. Max Pooling (2x2)
// ====================================

void maxpool2d(int H, int W, int channels,
               float input[H][W][channels],
               float output[H/2][W/2][channels]) {
    for (int h = 0; h < H / 2; h++) {
        for (int w = 0; w < W / 2; w++) {
            for (int c = 0; c < channels; c++) {
                float max_val = -1e9;
                for (int ph = 0; ph < 2; ph++) {
                    for (int pw = 0; pw < 2; pw++) {
                        int ih = h * 2 + ph;
                        int iw = w * 2 + pw;
                        if (input[ih][iw][c] > max_val)
                            max_val = input[ih][iw][c];
                    }
                }
                output[h][w][c] = max_val;
            }
        }
    }
}

// ====================================
// 5. GAP + Fully Connected
// ====================================

void global_avg_pool(int H, int W, int channels,
                     float input[H][W][channels],
                     float output[channels]) {
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                sum += input[h][w][c];
        output[c] = sum / (H * W);
    }
}

void fully_connected(int in_size, int out_size,
                     float input[in_size],
                     float output[out_size],
                     const float weights[out_size][in_size],
                     const float bias[out_size]) {
    for (int o = 0; o < out_size; o++) {
        float sum = bias[o];
        for (int i = 0; i < in_size; i++) {
            sum += input[i] * weights[o][i];
        }
        output[o] = sum;
    }
}

int argmax(int len, float input[len]) {
    int max_idx = 0;
    float max_val = input[0];
    for (int i = 1; i < len; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// ====================================
// 6. Main Inference
// ====================================

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <image_bin_file>\n", argv[0]);
        return -1;
    }

    size_t num_elements = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS;
    float *input_image = (float *)malloc(num_elements * sizeof(float));
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) {
        printf("Error: Could not open file %s\n", argv[1]);
        return -1;
    }
    fread(input_image, sizeof(float), num_elements, fp);
    fclose(fp);

    printf("Loaded image: %dx%d, channels=%d\n", IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS);

    // Allocate feature maps dynamically
    static float dw_out1[IMG_HEIGHT][IMG_WIDTH][IMG_CHANNELS];
    static float pw_out1[IMG_HEIGHT][IMG_WIDTH][BLOCK1_OUT];
    static float pool1[IMG_HEIGHT/2][IMG_WIDTH/2][BLOCK1_OUT];

    static float dw_out2[IMG_HEIGHT/2][IMG_WIDTH/2][BLOCK1_OUT];
    static float pw_out2[IMG_HEIGHT/2][IMG_WIDTH/2][BLOCK2_OUT];
    static float pool2[IMG_HEIGHT/4][IMG_WIDTH/4][BLOCK2_OUT];

    static float dw_out3[IMG_HEIGHT/4][IMG_WIDTH/4][BLOCK2_OUT];
    static float pw_out3[IMG_HEIGHT/4][IMG_WIDTH/4][BLOCK3_OUT];
    static float pool3[IMG_HEIGHT/8][IMG_WIDTH/8][BLOCK3_OUT];

    float gap_out[BLOCK3_OUT];
    float fc_out[NUM_CLASSES];

    clock_t start = clock();

    // Block 1
    depthwise_conv2d(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
                     (float (*)[IMG_WIDTH][IMG_CHANNELS])input_image,
                     dw_out1, conv1_dw_weight);
    pointwise_conv2d(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, BLOCK1_OUT,
                     dw_out1, pw_out1, conv1_pw_weight);
    batch_norm_relu(IMG_HEIGHT, IMG_WIDTH, BLOCK1_OUT,
                    pw_out1, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var);
    maxpool2d(IMG_HEIGHT, IMG_WIDTH, BLOCK1_OUT, pw_out1, pool1);

    // Block 2
    depthwise_conv2d(IMG_HEIGHT/2, IMG_WIDTH/2, BLOCK1_OUT, pool1, dw_out2, conv2_dw_weight);
    pointwise_conv2d(IMG_HEIGHT/2, IMG_WIDTH/2, BLOCK1_OUT, BLOCK2_OUT, dw_out2, pw_out2, conv2_pw_weight);
    batch_norm_relu(IMG_HEIGHT/2, IMG_WIDTH/2, BLOCK2_OUT, pw_out2, bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var);
    maxpool2d(IMG_HEIGHT/2, IMG_WIDTH/2, BLOCK2_OUT, pw_out2, pool2);

    // Block 3
    depthwise_conv2d(IMG_HEIGHT/4, IMG_WIDTH/4, BLOCK2_OUT, pool2, dw_out3, conv3_dw_weight);
    pointwise_conv2d(IMG_HEIGHT/4, IMG_WIDTH/4, BLOCK2_OUT, BLOCK3_OUT, dw_out3, pw_out3, conv3_pw_weight);
    batch_norm_relu(IMG_HEIGHT/4, IMG_WIDTH/4, BLOCK3_OUT, pw_out3, bn3_weight, bn3_bias, bn3_running_mean, bn3_running_var);
    maxpool2d(IMG_HEIGHT/4, IMG_WIDTH/4, BLOCK3_OUT, pw_out3, pool3);

    // GAP + FC
    global_avg_pool(IMG_HEIGHT/8, IMG_WIDTH/8, BLOCK3_OUT, pool3, gap_out);
    fully_connected(BLOCK3_OUT, NUM_CLASSES, gap_out, fc_out, fc_weight, fc_bias);

    int predicted = argmax(NUM_CLASSES, fc_out);
    const char *classes[] = {"Bird", "Drone"};
    printf("Prediction: %s\n", classes[predicted]);

    double total_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("Total Inference Time: %.4f s\n", total_time);

    free(input_image);
    return 0;
}
