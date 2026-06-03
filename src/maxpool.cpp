#include "yolo_utils.h"

void maxpool2d(
    const data_t *input, // flattened [Ci * Hi * Wi]
    data_t *output, // flattened [Ci * Ho * Wo]
    int Ci, int Hi, int Wi,
    int Ho, int Wo,
    int K = KERNEL_SIZE,
    int S = STRIDE,
    int P = PADDING
) {
    Loop_Ci: for (int ci = 0; ci < Ci; ci++) {
        Loop_Ho: for (int ho = 0; ho < Ho; ho++) {
            Loop_Wo: for (int wo = 0; wo < Wo; wo++) {
                #pragma HLS PIPELINE II=1

                int first_hi = ho * S - P;
                int first_wi = wo * S - P;
                data_t max_val = input[ci * (Hi * Wi) + first_hi * Wi + first_wi];

                Loop_K: for (int k = 0; k < K * K; k++) {
                    #pragma HLS UNROLL 
                    int kh = k / K;
                    int kw = k % K;
                    int hi = first_hi + kh;
                    int wi = first_wi + kw;

                    if (hi >= 0 && hi < Hi && wi >= 0 && wi < Wi) {
                        data_t val = input[ci * (Hi * Wi) + hi * Wi + wi];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }

                int out_idx = ci * (Ho * Wo) + ho * Wo + wo;
                output[out_idx] = max_val;
            }
        }
    }
} 