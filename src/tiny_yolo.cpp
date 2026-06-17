#include "yolo_utils.h"

#define INPUT_CHANNELS 3
#define INPUT_HEIGHT 64
#define INPUT_WIDTH 64

void tiny_yolo_lite(
	const data_t input[3][64][64],
	const data_t w1[8][3][3][3],
	const data_t b1[8],
	const data_t w2[16][8][3][3],
	const data_t b2[16],
	const data_t w3[32][16][3][3],
	const data_t b3[32],
	const data_t wh[30][32][1][1],
	const data_t bh[30],
	data_t output[30][12][12]
) {
    static data_t buffer_A[8][62][62];
    static data_t buffer_B[8][31][31];
    static data_t buffer_C[16][29][29];
    static data_t buffer_D[16][14][14];
    static data_t buffer_E[32][12][12];

    // LAYER 1: Conv2D + ReLu, Image -> buffer_A
    conv2d<3, 64, 64, 8, 62, 62, 3, 1, 0, true>(input, w1, b1, buffer_A);
    maxpool2d<8, 62, 62, 31, 31, 2, 2, 0>(buffer_A, buffer_B);

    // LAYER 2:
    conv2d<8, 31, 31, 16, 29, 29, 3, 1, 0, true>(buffer_B, w2, b2, buffer_C);
    maxpool2d<16, 29, 29, 14, 14, 2, 2, 0>(buffer_C, buffer_D);

    // LAYER 3:
    conv2d<16, 14, 14, 32, 12, 12, 3, 1, 0, true>(buffer_D, w3, b3, buffer_E);

    // HEAD: Conv2D 1x1, buffer_A -> local_output
    conv2d<32, 12, 12, 30, 12, 12, 1, 1, 0, false>(buffer_E, wh, bh, output);
}
