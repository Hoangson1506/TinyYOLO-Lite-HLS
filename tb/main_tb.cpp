#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include "../src/yolo_utils.h"

extern void tiny_yolo_lite(
    const data_t *input_image,
    const data_t *w_conv1, const data_t *b_conv1,
    const data_t *w_conv2, const data_t *b_conv2,
    const data_t *w_conv3, const data_t *b_conv3,
    const data_t *w_head,  const data_t *b_head,
    data_t *output_result
);

template<typename T>
bool load_data_from_txt(const std::string& filepath, T* array, int size) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filepath << std::endl;
        return false;
    }

    float val;
    for (int i = 0; i < size; i++) {
        if (!(file >> val)) {
            std::cerr << "Error: Can not read data at index " << i << " in file: " << filepath << std::endl;
            return false;
        }
        array[i] = (T)val;
    }

    file.close();
    return true;
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << " START C SIMULATION: TINY-YOLO LITE" << std::endl;
    std::cout << "===========================================" << std::endl;

    const int INPUT_SIZE = 3 * 64 * 64;
    const int OUTPUT_SIZE = 30 * 12 * 12;

    const int W1_SIZE = 8 * 3 * 3 * 3; // Conv1: CO=8, CI=3, K=3
    const int B1_SIZE = 8;
    const int W2_SIZE = 16 * 8 * 3 * 3; // Conv2: CO=16, CI=8, K=3
    const int B2_SIZE = 16;
    const int W3_SIZE = 32 * 16 * 3 * 3; // Conv3: CO=32, CI=16, K=3
    const int B3_SIZE = 32;
    const int W_HEAD_SIZE = 30 * 32 * 1 * 1; // Head: CO=30, CI=32, K=1
    const int B_HEAD_SIZE = 30;

    std::vector<data_t> input_image(INPUT_SIZE);
    std::vector<data_t> output_result(OUTPUT_SIZE);
    std::vector<data_t> golden_output(OUTPUT_SIZE); // raw float from PyTorch for comparison

    std::vector<data_t> w_conv1(W1_SIZE);
    std::vector<data_t> b_conv1(B1_SIZE);
    std::vector<data_t> w_conv2(W2_SIZE);
    std::vector<data_t> b_conv2(B2_SIZE);  
    std::vector<data_t> w_conv3(W3_SIZE);
    std::vector<data_t> b_conv3(B3_SIZE);
    std::vector<data_t> w_head(W_HEAD_SIZE);
    std::vector<data_t> b_head(B_HEAD_SIZE);

    std::cout << "Loading input and weights from text files..." << std::endl;
    bool load_ok = true;
    load_ok &= load_data_from_txt("data/inputs/dummy_input.txt", input_image.data(), INPUT_SIZE);

    load_ok &= load_data_from_txt("data/weights/conv1_weight.txt", w_conv1.data(), W1_SIZE);
    load_ok &= load_data_from_txt("data/weights/conv1_bias.txt", b_conv1.data(), B1_SIZE);
    load_ok &= load_data_from_txt("data/weights/conv2_weight.txt", w_conv2.data(), W2_SIZE);
    load_ok &= load_data_from_txt("data/weights/conv2_bias.txt", b_conv2.data(), B2_SIZE);
    load_ok &= load_data_from_txt("data/weights/conv3_weight.txt", w_conv3.data(), W3_SIZE);
    load_ok &= load_data_from_txt("data/weights/conv3_bias.txt", b_conv3.data(), B3_SIZE);
    load_ok &= load_data_from_txt("data/weights/detection_head_weight.txt", w_head.data(),  W_HEAD_SIZE);
    load_ok &= load_data_from_txt("data/weights/detection_head_bias.txt", b_head.data(),  B_HEAD_SIZE);

    if (!load_ok) {
        std::cerr << "Error: Failed to load all data. Exiting." << std::endl;
        return -1;
    }

    std::cout << "Running Tiny-YOLO Lite C Simulation..." << std::endl;
    tiny_yolo_lite(input_image.data(),
				   w_conv1.data(), b_conv1.data(),
				   w_conv2.data(), b_conv2.data(),
				   w_conv3.data(), b_conv3.data(),
				   w_head.data(),  b_head.data(),
				   output_result.data());
    std::cout << "Simulation completed." << std::endl;
    std::cout << "Comparing output with golden reference..." << std::endl;
    if (!load_data_from_txt("data/outputs/dummy_output.txt", golden_output.data(), OUTPUT_SIZE)) {
        std::cerr << "Error: Failed to load golden output for comparison." << std::endl;
        return -1;
    }

    int mismatch_count = 0;
    float max_error = 0.0f;
    float mse = 0.0f;

    const float TOLERANCE = 0.05f; // Tolerance for considering a match

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float diff = std::abs((float)output_result[i] - (float)golden_output[i]);
        mse += diff * diff;
        if (diff > TOLERANCE) {
            mismatch_count++;
            if (diff > max_error) {
                max_error = diff;
            }
            if (mismatch_count <= 10) { // Print first 10 mismatches
                std::cout << "Mismatch at index " << i 
                          << ": Output=" << (float)output_result[i] 
                          << ", Golden=" << (float)golden_output[i] 
                          << ", Diff=" << diff << std::endl;
            }
        }
    }

    mse /= OUTPUT_SIZE;

    std::cout << "Total mismatches: " << mismatch_count << " out of " << OUTPUT_SIZE << std::endl;
    std::cout << "MSE: " << mse << std::endl;
    std::cout << "MAE: " << max_error << std::endl;

    if (mismatch_count == 0) {
    	std::cout << "PASS: All outputs match the golden reference within the tolerance." << std::endl;
		return 0;
	} else {
		std::cout << "FAIL: There are some mismatches. Please check the above details for debugging." << std::endl;
		return 1;
	}
}
