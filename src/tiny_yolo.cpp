#include "yolo_utils.h"

#define INPUT_CHANNELS 3
#define INPUT_HEIGHT 64
#define INPUT_WIDTH 64

#define MAX_BUFFER_SIZE 32000

void tiny_yolo_lite(
    const data_t *input, // [3 * 64 * 64]
    const data_t *conv1_weights, const data_t *conv1_bias,
    const data_t *conv2_weights, const data_t *conv2_bias,
    const data_t *conv3_weights, const data_t *conv3_bias,
    const data_t *w_head, const data_t *b_head,
    data_t *output // [30 * 8 * 8]
) {
    // ==========================================
    // CHỈ ĐẠO HLS GIAO TIẾP PHẦN CỨNG (INTERFACES)
    // ==========================================
    // Sử dụng AXI4 Master (m_axi) để phần cứng có thể tự động DMA đọc/ghi RAM (DDR) bên ngoài
    #pragma HLS INTERFACE m_axi port=input_image offset=slave bundle=gmem_in
    #pragma HLS INTERFACE m_axi port=output_result offset=slave bundle=gmem_out
    
    // Gộp tất cả weights vào chung một bus để tiết kiệm cổng giao tiếp
    #pragma HLS INTERFACE m_axi port=conv1_weights offset=slave bundle=gmem_weights
    #pragma HLS INTERFACE m_axi port=conv1_bias offset=slave bundle=gmem_weights
    #pragma HLS INTERFACE m_axi port=conv2_weights offset=slave bundle=gmem_weights
    #pragma HLS INTERFACE m_axi port=conv2_bias offset=slave bundle=gmem_weights
    #pragma HLS INTERFACE m_axi port=conv3_weights offset=slave bundle=gmem_weights
    #pragma HLS INTERFACE m_axi port=conv3_bias offset=slave bundle=gmem_weights
    #pragma HLS INTERFACE m_axi port=w_head offset=slave bundle=gmem_weights
    #pragma HLS INTERFACE m_axi port=b_head offset=slave bundle=gmem_weights

    // Sử dụng AXI4-Lite (s_axilite) làm các thanh ghi điều khiển từ CPU (ARM)
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS

    // ==========================================
    // KHAI BÁO BỘ NHỚ TRUNG GIAN (PING-PONG BUFFERS)
    // ==========================================
    // Từ khóa 'static' đảm bảo HLS tổng hợp các mảng này thành BRAM vật lý trên FPGA
    // thay vì tạo ra các thanh ghi (registers) rời rạc làm cháy chip.
    static data_t buffer_A[MAX_BUFFER_SIZE];
    static data_t buffer_B[MAX_BUFFER_SIZE];

    // LAYER 1: Conv2D + ReLu, Image -> buffer_A
    conv2d(input, conv1_weights, conv1_bias, buffer_A,
           3, 64, 64,  // Input: CI, IH, IW
           8, 62, 62,  // Output: CO, OH, OW
           3, 1, 0,      // Kernel=3, Stride=1, Padding=0
           true);        // Dùng ReLU
    // buffer_A -> buffer_B maxpool
    maxpool2d(buffer_A, buffer_B,
              8, 62, 62,   // Input: CI, IH, IW
              31, 31,    // Output: OH, OW 
              2, 2, 0);  // Kernel: K, S, P

    // LAYER 2:
    conv2d(buffer_B, conv2_weights, conv2_bias, buffer_A,
           8, 31, 31,   // Input: CI, IH, IW
           16, 29, 29,  // Output: CO, OH, OW
           3, 1, 0,      // Kernel=3, Stride=1, Padding=0
           true);        // Dùng ReLU
    // buffer_A -> buffer_B maxpool
    maxpool2d(buffer_A, buffer_B,
              16, 29, 29,   // Input: CI, IH, IW
              14, 14,    // Output: OH, OW 
              2, 2, 0);  // Kernel: K, S, P
    
    // LAYER 3:
    conv2d(buffer_B, conv3_weights, conv3_bias, buffer_A,
           16, 14, 14,   // Input: CI, IH, IW
           32, 12, 12,   // Output: CO, OH, OW
           3, 1, 0,      // Kernel=3, Stride=1, Padding=0
           true);        // Dùng ReLU

    // HEAD: Conv2D 1x1, buffer_A -> output
    conv2d(buffer_A, w_head, b_head, output,
           32, 12, 12,   // Input: CI, IH, IW
           30, 12, 12,     // Output: CO, OH, OW
           1, 1, 0,      // Kernel=1, Stride=1, Padding=0
           false);       // Không dùng ReLU ở layer cuối
}