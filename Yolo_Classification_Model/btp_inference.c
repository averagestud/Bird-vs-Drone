/*
 * inference.c
 *
 * Implements the C inference logic for PatchClassifier6.
 * All helper functions and buffers are 'static' to this file.
 */

#include "inference.h"
#include <math.h>     // For expf()
#include <string.h>   // For memset()
#include <stdio.h>    // For printf (debugging)
#include <stdlib.h>   // For aligned_alloc, free

// ===================================================================
// 1. Static Intermediate Buffers
// ===================================================================
// We define static buffers for each layer's output to avoid stack
// overflow and dynamic allocation.

// Output dimensions calculated from the PyTorch model
// H_out = floor((H_in + 2*P - K) / S) + 1

static float stem_out[STEM_C_OUT][81][81];           // K=3, S=2, P=1 -> (162+2-3)/2 + 1 = 81
static float l1_dw_out[L1_C_MID][41][41];            // K=3, S=2, P=1 -> (81+2-3)/2 + 1 = 41
static float l1_out[L1_C_OUT][41][41];               // K=1, S=1, P=0 -> (41+0-1)/1 + 1 = 41
static float l2_dw_out[L2_C_MID][21][21];            // K=3, S=2, P=1 -> (41+2-3)/2 + 1 = 21
static float l2_out[L2_C_OUT][21][21];               // K=1, S=1, P=0 -> (21+0-1)/1 + 1 = 21
static float l3_dw_out[L3_C_MID][11][11];            // K=3, S=2, P=1 -> (21+2-3)/2 + 1 = 11
static float l3_out[L3_C_OUT][11][11];               // K=1, S=1, P=0 -> (11+0-1)/1 + 1 = 11
static float l4_dw_out[L4_C_MID][6][6];              // K=3, S=2, P=1 -> (11+2-3)/2 + 1 = 6
static float l4_out[L4_C_OUT][6][6];                 // K=1, S=1, P=0 -> (6+0-1)/1 + 1 = 6
static float l5_dw_out[L5_C_MID][6][6];              // K=3, S=1, P=1 -> (6+2-3)/1 + 1 = 6
static float l5_out[L5_C_OUT][6][6];                 // K=1, S=1, P=0 -> (6+0-1)/1 + 1 = 6

static float pooled_out[FC_PATCH_IN];                // [256]
static float fc_patch_out[FC_PATCH_OUT];             // [32]

// Buffers for aggregation
static float all_patch_features[NUM_PATCHES][FC_PATCH_OUT]; // [16][32]
static float mean_features[FC_PATCH_OUT];                   // [32]
static float final_logit[FC_FINAL_OUT];                     // [1]


// ===================================================================
// 2. Helper Functions (NN Primitives)
// ===================================================================

static float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// In-place ReLU activation
static void relu(float* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}

// Helper to get input value with zero-padding
static float get_padded_in(const float* in, int c, int h, int w, int C_IN, int H_IN, int W_IN) {
    if (h < 0 || h >= H_IN || w < 0 || w >= W_IN) {
        return 0.0f;
    }
    // Assumes layout [C][H][W]
    return in[c * H_IN * W_IN + h * W_IN + w];
}

/**
 * @brief General 2D convolution with padding, stride, and groups.
 * Assumes data is laid out [C][H][W].
 */
static void conv2d(
    const float* in,  // Input: [C_IN, H_IN, W_IN] (flat)
    float* out,       // Output: [C_OUT, H_OUT, W_OUT] (flat)
    const float* w,   // Weights: [C_OUT, C_IN/G, K, K] (flat)
    const float* b,   // Bias: [C_OUT]
    int H_IN, int W_IN, int C_IN,
    int H_OUT, int W_OUT, int C_OUT,
    int K, int P, int S, int G
) {
    int C_IN_PER_GROUP = C_IN / G;
    
    // For each output channel
    for (int co = 0; co < C_OUT; co++) {
        int g = co / (C_OUT / G); // Current group
        
        // For each output pixel
        for (int oh = 0; oh < H_OUT; oh++) {
            for (int ow = 0; ow < W_OUT; ow++) {
                
                float sum = b[co];
                int h_in_start = oh * S - P;
                int w_in_start = ow * S - P;
                
                // Apply kernel
                for (int ci_g = 0; ci_g < C_IN_PER_GROUP; ci_g++) {
                    int ci = g * C_IN_PER_GROUP + ci_g; // Actual input channel
                    
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int h_here = h_in_start + kh;
                            int w_here = w_in_start + kw;
                            
                            float in_val = get_padded_in(in, ci, h_here, w_here, C_IN, H_IN, W_IN);
                            
                            // Weight index: [co, ci_g, kh, kw]
                            int w_idx = co * C_IN_PER_GROUP * K * K + 
                                        ci_g * K * K + 
                                        kh * K + 
                                        kw;
                                        
                            sum += in_val * w[w_idx];
                        }
                    }
                }
                out[co * H_OUT * W_OUT + oh * W_OUT + ow] = sum;
            }
        }
    }
}

// Flattens [C, H, W] to [C] by averaging
static void adaptive_avg_pool2d(
    const float* in,  // Input: [C, H, W] (flat)
    float* out,       // Output: [C] (flat)
    int C, int H, int W
) {
    double inv_size = 1.0 / (H * W);
    for (int c = 0; c < C; c++) {
        double sum = 0.0;
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                sum += in[c * H * W + h * W + w];
            }
        }
        out[c] = (float)(sum * inv_size);
    }
}

// Fully connected layer
static void linear(
    const float* in,  // Input: [IN_F]
    float* out,       // Output: [OUT_F]
    const float* w,   // Weights: [OUT_F, IN_F]
    const float* b,   // Bias: [OUT_F]
    int IN_F, int OUT_F
) {
    for (int o = 0; o < OUT_F; o++) {
        float sum = b[o];
        for (int i = 0; i < IN_F; i++) {
            // Weight index: [o, i]
            sum += in[i] * w[o * IN_F + i];
        }
        out[o] = sum;
    }
}

// ===================================================================
// 3. Per-Patch CNN Backbone
// ===================================================================

/**
 * @brief Runs one patch through the CNN backbone (stem to fc_patch).
 * @param patch_in      Pointer to a single patch [1][162][162]
 * @param features_out  Pointer to the output buffer [32]
 */
static void run_patch_cnn(const float* patch_in, float* features_out) {
    // ---- Stem ----
    // In: [1, 162, 162], Out: [16, 81, 81]
    conv2d((const float*)patch_in, (float*)stem_out, (const float*)stem_conv_w, stem_conv_b,
           162, 162, STEM_C_IN, 81, 81, STEM_C_OUT, 3, 1, 2, 1);
    relu((float*)stem_out, 16*81*81);

    // ---- Layer 1 ----
    // In: [16, 81, 81], Out: [32, 41, 41]
    conv2d((const float*)stem_out, (float*)l1_dw_out, (const float*)l1_dw_conv_w, l1_dw_conv_b,
           81, 81, L1_C_IN, 41, 41, L1_C_MID, 3, 1, 2, L1_C_IN); // Depthwise (G=C_IN)
    relu((float*)l1_dw_out, 16*41*41);
    conv2d((const float*)l1_dw_out, (float*)l1_out, (const float*)l1_pw_conv_w, l1_pw_conv_b,
           41, 41, L1_C_MID, 41, 41, L1_C_OUT, 1, 0, 1, 1);      // Pointwise (G=1, K=1)
    relu((float*)l1_out, 32*41*41);

    // ---- Layer 2 ----
    // In: [32, 41, 41], Out: [64, 21, 21]
    conv2d((const float*)l1_out, (float*)l2_dw_out, (const float*)l2_dw_conv_w, l2_dw_conv_b,
           41, 41, L2_C_IN, 21, 21, L2_C_MID, 3, 1, 2, L2_C_IN);
    relu((float*)l2_dw_out, 32*21*21);
    conv2d((const float*)l2_dw_out, (float*)l2_out, (const float*)l2_pw_conv_w, l2_pw_conv_b,
           21, 21, L2_C_MID, 21, 21, L2_C_OUT, 1, 0, 1, 1);
    relu((float*)l2_out, 64*21*21);

    // ---- Layer 3 ----
    // In: [64, 21, 21], Out: [128, 11, 11]
    conv2d((const float*)l2_out, (float*)l3_dw_out, (const float*)l3_dw_conv_w, l3_dw_conv_b,
           21, 21, L3_C_IN, 11, 11, L3_C_MID, 3, 1, 2, L3_C_IN);
    relu((float*)l3_dw_out, 64*11*11);
    conv2d((const float*)l3_dw_out, (float*)l3_out, (const float*)l3_pw_conv_w, l3_pw_conv_b,
           11, 11, L3_C_MID, 11, 11, L3_C_OUT, 1, 0, 1, 1);
    relu((float*)l3_out, 128*11*11);

    // ---- Layer 4 ----
    // In: [128, 11, 11], Out: [256, 6, 6]
    conv2d((const float*)l3_out, (float*)l4_dw_out, (const float*)l4_dw_conv_w, l4_dw_conv_b,
           11, 11, L4_C_IN, 6, 6, L4_C_MID, 3, 1, 2, L4_C_IN);
    relu((float*)l4_dw_out, 128*6*6);
    conv2d((const float*)l4_dw_out, (float*)l4_out, (const float*)l4_pw_conv_w, l4_pw_conv_b,
           6, 6, L4_C_MID, 6, 6, L4_C_OUT, 1, 0, 1, 1);
    relu((float*)l4_out, 256*6*6);

    // ---- Layer 5 ----
    // In: [256, 6, 6], Out: [256, 6, 6]
    conv2d((const float*)l4_out, (float*)l5_dw_out, (const float*)l5_dw_conv_w, l5_dw_conv_b,
           6, 6, L5_C_IN, 6, 6, L5_C_MID, 3, 1, 1, L5_C_IN); // Stride = 1
    relu((float*)l5_dw_out, 256*6*6);
    conv2d((const float*)l5_dw_out, (float*)l5_out, (const float*)l5_pw_conv_w, l5_pw_conv_b,
           6, 6, L5_C_MID, 6, 6, L5_C_OUT, 1, 0, 1, 1);
    relu((float*)l5_out, 256*6*6);

    // ---- Pooling ----
    // In: [256, 6, 6], Out: [256]
    adaptive_avg_pool2d((const float*)l5_out, (float*)pooled_out,
                        L5_C_OUT, 6, 6);

    // ---- FC Patch ----
    // In: [256], Out: [32]
    // Note: No ReLU after this layer in the PyTorch model
    linear((const float*)pooled_out, features_out, (const float*)fc_patch_w, fc_patch_b,
           FC_PATCH_IN, FC_PATCH_OUT);
}


// ===================================================================
// 4. Public Inference Function
// ===================================================================

float predict_drone_prob(const float* all_patches_in) {
    
    // --- 1. Run CNN backbone on all 16 patches ---
    for (int p = 0; p < NUM_PATCHES; p++) {
        // Calculate the starting pointer for the p-th patch
        const float* current_patch = all_patches_in + (p * STEM_C_IN * PATCH_H_IN * PATCH_W_IN);
        
        // Run the CNN and store the [32] feature vector
        run_patch_cnn(current_patch, all_patch_features[p]);
    }

    // --- 2. Pool (mean) the patch features ---
    memset(mean_features, 0, sizeof(mean_features));
    for (int p = 0; p < NUM_PATCHES; p++) {
        for (int f = 0; f < FC_PATCH_OUT; f++) {
            mean_features[f] += all_patch_features[p][f];
        }
    }
    float inv_patches = 1.0f / NUM_PATCHES;
    for (int f = 0; f < FC_PATCH_OUT; f++) {
        mean_features[f] *= inv_patches;
    }

    // --- 3. Run final FC layer ---
    // In: [32], Out: [1]
    linear((const float*)mean_features, (float*)final_logit, (const float*)fc_final_w, fc_final_b,
           FC_FINAL_IN, FC_FINAL_OUT);

    // --- 4. Apply sigmoid to get probability ---
    return sigmoidf(final_logit[0]);
}

/*
 * main.c
 *
 * Example program to test the PatchClassifier6 C inference.
 */

/*
 * main.c
 *
 * Example program to load a RAW 640x640 1-channel .bin file,
 * process it into 16 padded patches, and run inference.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "inference.h" // Your inference functions
#include "btp_class_weights.h" // Your weight dimensions

// --- Model Parameters (from your Python) ---
#define IMG_SIZE 640
#define PATCH_GRID 4
#define PATCH_SIZE 160 // (IMG_SIZE / PATCH_GRID)
#define PAD 1

// Define the full input size
#define TOTAL_INPUT_FLOATS (NUM_PATCHES * STEM_C_IN * PATCH_H_IN * PATCH_W_IN)
// We expect a raw file with exactly 640*640 bytes
#define EXPECTED_FILE_SIZE (IMG_SIZE * IMG_SIZE)

/**
 * @brief get_src_coord
 * Helper to implement BORDER_REFLECT padding.
 */
static inline int get_src_coord(int coord, int max_dim) {
    if (coord < 0) {
        return 1; // Reflect -1 to 1
    }
    if (coord >= max_dim) {
        return max_dim - 2; // Reflect 160 to 158 (max_dim-2)
    }
    return coord;
}

int main(int argc, char *argv[]) {
    
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path_to_640x640_raw.bin>\n", argv[0]);
        return 1;
    }
    
    char* filename = argv[1];
    printf("Starting C inference for: %s\n", filename);

    // 1. Allocate memory for the final 16-patch buffer
    float* all_patches_buffer = (float*)malloc(TOTAL_INPUT_FLOATS * sizeof(float));
    if (all_patches_buffer == NULL) {
        fprintf(stderr, "Failed to allocate memory for patch buffer\n");
        return 1;
    }

    // 2. Load the raw .bin file
    FILE *f = fopen(filename, "rb");
    if (f == NULL) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        free(all_patches_buffer);
        return 1;
    }

    // Check file size to ensure it matches our 640x640 assumption
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size != EXPECTED_FILE_SIZE) {
        fprintf(stderr, "Error: File size mismatch. Expected %d bytes (640x640), but got %ld bytes.\n",
                EXPECTED_FILE_SIZE, file_size);
        fclose(f);
        free(all_patches_buffer);
        return 1;
    }

    // Allocate buffer for the raw image data
    unsigned char *img_data = (unsigned char*)malloc(EXPECTED_FILE_SIZE);
    if (img_data == NULL) {
        fprintf(stderr, "Failed to allocate memory for image buffer\n");
        fclose(f);
        free(all_patches_buffer);
        return 1;
    }

    // Read the entire file into the buffer
    size_t bytes_read = fread(img_data, 1, EXPECTED_FILE_SIZE, f);
    if (bytes_read != EXPECTED_FILE_SIZE) {
        fprintf(stderr, "Error: Failed to read %d bytes from file.\n", EXPECTED_FILE_SIZE);
        fclose(f);
        free(img_data);
        free(all_patches_buffer);
        return 1;
    }
    fclose(f); // Done with the file

    printf("Loaded %ld bytes (640x640). Processing patches...\n", file_size);

    // 3. Process the image into 16 padded patches
    // (This logic is identical to the previous version)
    for (int p = 0; p < NUM_PATCHES; p++) {
        int p_row = p / PATCH_GRID;
        int p_col = p % PATCH_GRID;

        // Calculate start coordinates of the 160x160 patch
        int patch_start_x = p_col * PATCH_SIZE;
        int patch_start_y = p_row * PATCH_SIZE;

        // Get the pointer to the destination buffer for this *single* patch
        float* patch_dest_ptr = all_patches_buffer + (p * PATCH_H_IN * PATCH_W_IN);

        // Fill the 162x162 destination patch
        for (int y_dst = 0; y_dst < PATCH_H_IN; y_dst++) {
            for (int x_dst = 0; x_dst < PATCH_W_IN; x_dst++) {
                
                int x_src_unpadded = x_dst - PAD;
                int y_src_unpadded = y_dst - PAD;

                int x_src_padded = get_src_coord(x_src_unpadded, PATCH_SIZE);
                int y_src_padded = get_src_coord(y_src_unpadded, PATCH_SIZE);

                int x_src_full = patch_start_x + x_src_padded;
                int y_src_full = patch_start_y + y_src_padded;
                
                // Get the pixel (unsigned char)
                unsigned char pixel_val = img_data[y_src_full * IMG_SIZE + x_src_full];

                // Normalize (0-255 -> 0.0-1.0) and write to destination
                patch_dest_ptr[y_dst * PATCH_H_IN + x_dst] = (float)pixel_val / 255.0f;
            }
        }
    }
    
    // We are done with the raw image data buffer
    free(img_data);
    printf("...Patch processing complete.\n");

    // 4. Run inference
    printf("Running inference...\n");
    float probability = predict_drone_prob(all_patches_buffer);
    printf("...Inference complete.\n");

    // 5. Print the result
    printf("========================================\n");
    printf("Final Drone Probability: %.6f\n", probability);
    printf("========================================\n");

    // 6. Clean up
    free(all_patches_buffer);
    return 0;
}