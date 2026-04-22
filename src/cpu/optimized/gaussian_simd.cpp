#include "filters.h"
#include "utils.h"
#include <iostream>
#include <cmath>
#include <immintrin.h>  // AVX2 intrinsics
#include <algorithm>

// Build 1D Gaussian kernel
static std::vector<double> build_kernel(int kernel_size, double sigma) {
    std::vector<double> kernel(kernel_size);
    int half = kernel_size / 2;
    double sum = 0.0;

    for (int i = 0; i < kernel_size; i++) {
        int x = i - half;
        kernel[i] = std::exp(-(x * x) / (2.0 * sigma * sigma));
        sum += kernel[i];
    }

    // Normalize
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }

    return kernel;
}

// Convert kernel to float for SIMD
static std::vector<float> kernel_to_float(const std::vector<double>& kernel) {
    std::vector<float> fkernel(kernel.size());
    for (size_t i = 0; i < kernel.size(); i++) {
        fkernel[i] = static_cast<float>(kernel[i]);
    }
    return fkernel;
}

// Horizontal pass with SIMD (AVX2 - process 8 pixels at a time)
static void horizontal_pass_simd(const cv::Mat& src, cv::Mat& temp, 
                                  const std::vector<float>& kernel) {
    int kernel_size = kernel.size();
    int half = kernel_size / 2;
    int rows = src.rows;
    int cols = src.cols;

    for (int r = 0; r < rows; r++) {
        const uchar* src_row = src.ptr<uchar>(r);
        float* temp_row = temp.ptr<float>(r);

        for (int c = 0; c < cols; c += 8) {
            __m256 result = _mm256_setzero_ps();

            // Convolution with kernel
            for (int k = -half; k <= half; k++) {
                // Handle boundary
                int cc_base = c + k;
                
                // Load 8 pixels (with boundary handling)
                float pixels[8];
                for (int i = 0; i < 8; i++) {
                    int cc = cc_base + i;
                    if (cc < 0) cc = 0;
                    if (cc >= cols) cc = cols - 1;
                    pixels[i] = static_cast<float>(src_row[cc]);
                }

                __m256 pixel_vec = _mm256_loadu_ps(pixels);
                __m256 kernel_val = _mm256_set1_ps(kernel[k + half]);
                result = _mm256_fmadd_ps(pixel_vec, kernel_val, result);
            }

            // Store result
            _mm256_storeu_ps(temp_row + c, result);
        }

        // Handle remaining pixels (cols % 8)
        for (int c = (cols / 8) * 8; c < cols; c++) {
            float val = 0.0f;
            for (int k = -half; k <= half; k++) {
                int cc = c + k;
                if (cc < 0) cc = 0;
                if (cc >= cols) cc = cols - 1;
                val += kernel[k + half] * src_row[cc];
            }
            temp_row[c] = val;
        }
    }
}

// Vertical pass with SIMD (AVX2)
static void vertical_pass_simd(const cv::Mat& temp, cv::Mat& dst,
                                const std::vector<float>& kernel) {
    int kernel_size = kernel.size();
    int half = kernel_size / 2;
    int rows = temp.rows;
    int cols = temp.cols;

    for (int c = 0; c < cols; c += 8) {
        for (int r = 0; r < rows; r++) {
            __m256 result = _mm256_setzero_ps();

            // Convolution with kernel
            for (int k = -half; k <= half; k++) {
                int rr_base = r + k;
                if (rr_base < 0) rr_base = 0;
                if (rr_base >= rows) rr_base = rows - 1;

                // Load 8 values from column
                float vals[8];
                for (int i = 0; i < 8; i++) {
                    int col = c + i;
                    if (col < cols) {
                        vals[i] = temp.at<float>(rr_base, col);
                    }
                }

                __m256 val_vec = _mm256_loadu_ps(vals);
                __m256 kernel_val = _mm256_set1_ps(kernel[k + half]);
                result = _mm256_fmadd_ps(val_vec, kernel_val, result);
            }

            // Store result
            for (int i = 0; i < 8; i++) {
                int col = c + i;
                if (col < cols) {
                    dst.at<uchar>(r, col) = static_cast<uchar>(
                        std::round(((float*)&result)[i])
                    );
                }
            }
        }
    }
}

cv::Mat gaussian_simd(const cv::Mat& src, int kernel_size, double sigma) {
    CV_Assert(src.type() == CV_8UC1);  // Grayscale only
    CV_Assert(kernel_size % 2 == 1);   // Must be odd

    int rows = src.rows;
    int cols = src.cols;

    std::vector<double> kernel = build_kernel(kernel_size, sigma);
    std::vector<float> fkernel = kernel_to_float(kernel);

    cv::Mat temp = cv::Mat::zeros(rows, cols, CV_32F);
    cv::Mat dst = cv::Mat::zeros(rows, cols, CV_8UC1);

    // Horizontal pass with SIMD
    horizontal_pass_simd(src, temp, fkernel);

    // Vertical pass with SIMD
    vertical_pass_simd(temp, dst, fkernel);

    return dst;
}

int main(int argc, char* argv[]) {
    std::string image_path = (argc > 1) ? argv[1] : "test_image.png";

    cv::Mat src;
    try {
        src = load_image(image_path);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << src.cols << "x" << src.rows << std::endl;

    // Warm up
    gaussian_simd(src, 15, 2.0);

    // Benchmark: run 5 times, take average
    const int RUNS = 5;
    double total_ms = 0.0;
    cv::Mat result;
    Timer timer;

    for (int i = 0; i < RUNS; i++) {
        timer.start();
        result = gaussian_simd(src, 15, 2.0);
        timer.stop();
        total_ms += timer.elapsed_ms();
    }

    double avg_ms = total_ms / RUNS;
    std::cout << "SIMD Gaussian (avg " << RUNS << " runs): " 
              << avg_ms << " ms" << std::endl;

    save_image("output_simd.png", result);
    std::cout << "Saved: output_simd.png" << std::endl;

    return 0;
}