// Phase 2.1: gaussian_kernel.cu
// Basic CUDA Gaussian blur kernel (compute time only, no memory optimization)

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>

// CUDA kernel: Horizontal pass
__global__ void horizontal_pass_kernel(const uchar* src, float* temp,
                                       int rows, int cols,
                                       const float* kernel, int kernel_size) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r >= rows || c >= cols) return;

    int half = kernel_size / 2;
    float val = 0.0f;

    for (int k = -half; k <= half; k++) {
        int cc = c + k;
        if (cc < 0) cc = 0;
        if (cc >= cols) cc = cols - 1;
        val += kernel[k + half] * src[r * cols + cc];
    }

    temp[r * cols + c] = val;
}

// CUDA kernel: Vertical pass
__global__ void vertical_pass_kernel(const float* temp, uchar* dst,
                                     int rows, int cols,
                                     const float* kernel, int kernel_size) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r >= rows || c >= cols) return;

    int half = kernel_size / 2;
    float val = 0.0f;

    for (int k = -half; k <= half; k++) {
        int rr = r + k;
        if (rr < 0) rr = 0;
        if (rr >= rows) rr = rows - 1;
        val += kernel[k + half] * temp[rr * cols + c];
    }

    dst[r * cols + c] = (uchar)roundf(val);
}

// Build 1D Gaussian kernel (on host)
std::vector<float> build_kernel(int kernel_size, double sigma) {
    std::vector<float> kernel(kernel_size);
    int half = kernel_size / 2;
    float sum = 0.0f;

    for (int i = 0; i < kernel_size; i++) {
        int x = i - half;
        kernel[i] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }

    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }

    return kernel;
}

cv::Mat gaussian_gpu_kernel(const cv::Mat& src, int kernel_size, double sigma) {
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(kernel_size % 2 == 1);

    int rows = src.rows;
    int cols = src.cols;
    size_t img_bytes = rows * cols;

    // Build kernel on host
    std::vector<float> h_kernel = build_kernel(kernel_size, sigma);

    // Allocate device memory
    uchar* d_src = nullptr;
    float* d_temp = nullptr;
    uchar* d_dst = nullptr;
    float* d_kernel = nullptr;

    cudaMalloc(&d_src, img_bytes);
    cudaMalloc(&d_temp, img_bytes * sizeof(float));
    cudaMalloc(&d_dst, img_bytes);
    cudaMalloc(&d_kernel, h_kernel.size() * sizeof(float));

    // Copy to device
    cudaMemcpy(d_src, src.data, img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), h_kernel.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    // Launch kernels
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);

    horizontal_pass_kernel<<<gridSize, blockSize>>>(d_src, d_temp, rows, cols,
                                                     d_kernel, kernel_size);
    cudaDeviceSynchronize();

    vertical_pass_kernel<<<gridSize, blockSize>>>(d_temp, d_dst, rows, cols,
                                                   d_kernel, kernel_size);
    cudaDeviceSynchronize();

    // Copy result back
    cv::Mat result(rows, cols, CV_8UC1);
    cudaMemcpy(result.data, d_dst, img_bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_src);
    cudaFree(d_temp);
    cudaFree(d_dst);
    cudaFree(d_kernel);

    return result;
}

int main(int argc, char* argv[]) {
    std::string image_path = (argc > 1) ? argv[1] : "test_image.png";

    cv::Mat src = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << src.cols << "x" << src.rows << std::endl;

    // Warm up
    gaussian_gpu_kernel(src, 15, 2.0);

    // Benchmark: 5 runs
    const int RUNS = 5;
    double total_ms = 0.0;
    cv::Mat result;

    for (int i = 0; i < RUNS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        result = gaussian_gpu_kernel(src, 15, 2.0);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        total_ms += duration.count();
    }

    double avg_ms = total_ms / RUNS;
    std::cout << "GPU Kernel Gaussian (avg " << RUNS << " runs): " 
              << avg_ms << " ms" << std::endl;

    cv::imwrite("output_gpu_kernel.png", result);
    std::cout << "Saved: output_gpu_kernel.png" << std::endl;

    return 0;
}