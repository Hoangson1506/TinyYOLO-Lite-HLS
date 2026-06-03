#include "yolo_utils.h"

void conv2d(
    const data_t *input, // flattened [Ci * Hi * Wi]
    const data_t *weights, // flattened [Co * Ci * K * K]
    const data_t *bias, // [Co]
    data_t *output, // flattened [Co * Ho * Wo]
    int Ci, int Hi, int Wi,
    int Co, int Ho, int Wo,
    int K, int S, int P,
    bool apply_relu = true
) {
    Loop_Co: for (int co = 0; co < Co; co++) {
        Loop_Ho: for (int ho = 0; ho < Ho; ho++) {
            Loop_Wo: for (int wo = 0; wo < Wo; wo++) {
                // --- CHỈ ĐẠO HLS (PRAGMAS) ---
                // Pipeline ở mức tính toán Pixel để tăng Throughput.
                // Vivado HLS sẽ cố gắng thực hiện phần thân vòng lặp này trong 1 Clock Cycle.
                #pragma HLS PIPELINE II=1
                mac_t acc = bias[co];

                Loop_Ci: for (int ci = 0; ci < Ci; ci++) {
                    // --- CHỈ ĐẠO HLS ---
                    // Duỗi vòng lặp CI để tính song song MAC cho nhiều Input Channel cùng lúc.
                    // Lưu ý: Cần kết hợp #pragma HLS ARRAY_PARTITION ở top-level cho mảng input/weights.
                    #pragma HLS UNROLL factor=4

                    Loop_K: for (int k = 0; k < K * K; k++) {
                        int kh = k / K;
                        int kw = k % K;
                        int hi = ho * S + kh - P;
                        int wi = wo * S + kw - P;

                        if (hi >= 0 && hi < Hi && wi >= 0 && wi < Wi) {
                            int in_idx = ci * (Hi * Wi) + hi * Wi + wi;
                            int w_idx = co * (Ci * K * K) +
                                        ci * (K * K) +
                                        kh * K + kw;
                            acc += input[in_idx] * weights[w_idx];
                        }
                    }
                }

                int out_idx = co * (Ho * Wo) + ho * Wo + wo;

                if (apply_relu) {
                    output[out_idx] = (acc > 0) ? (data_t)acc : (data_t)0;
                } else {
                    output[out_idx] = (data_t)acc;
                }
            }
        }
    }
}