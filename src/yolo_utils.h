#ifndef YOLO_UTILS_H
#define YOLO_UTILS_H

#include <ap_fixed.h>
#include <algorithm>

// Cấu hình Quantization: Tổng 16 bit, 6 bit phần nguyên, 10 bit phần thập phân.
// Có thể tinh chỉnh <16, 6> xuống <8, 4> tùy vào kết quả C Simulation.
typedef ap_fixed<16, 6> data_t; 
typedef ap_fixed<32, 12> mac_t; // Kiểu dữ liệu lớn hơn cho bộ Accumulator để tránh Overflow

#define KERNEL_SIZE 3
#define STRIDE 1
#define PADDING 0

#endif