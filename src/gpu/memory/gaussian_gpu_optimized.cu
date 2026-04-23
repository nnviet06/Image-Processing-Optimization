// Phase 2.2: gaussian_gpu_optimized.cu
// GPU with pinned host memory + CUDA streams

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>

// Same kernels as Phase 2.1
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

cv::Mat gaussian_gpu_optimized(const cv::Mat& src, int kernel_size, double sigma) {
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(kernel_size % 2 == 1);

    int rows = src.rows;
    int cols = src.cols;
    size_t img_bytes = rows * cols;

    std::vector<float> h_kernel = build_kernel(kernel_size, sigma);

    // Allocate PINNED host memory (faster PCIe transfers)
    uchar* h_src_pinned = nullptr;
    uchar* h_dst_pinned = nullptr;
    cudaMallocHost(&h_src_pinned, img_bytes);
    cudaMallocHost(&h_dst_pinned, img_bytes);

    // Copy source to pinned memory
    memcpy(h_src_pinned, src.data, img_bytes);

    // Device memory
    uchar* d_src = nullptr;
    float* d_temp = nullptr;
    uchar* d_dst = nullptr;
    float* d_kernel = nullptr;

    cudaMalloc(&d_src, img_bytes);
    cudaMalloc(&d_temp, img_bytes * sizeof(float));
    cudaMalloc(&d_dst, img_bytes);
    cudaMalloc(&d_kernel, h_kernel.size() * sizeof(float));

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Async copy (with stream)
    cudaMemcpyAsync(d_src, h_src_pinned, img_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_kernel, h_kernel.data(), h_kernel.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // Launch kernels
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);

    horizontal_pass_kernel<<<gridSize, blockSize, 0, stream>>>(d_src, d_temp, rows, cols,
                                                                d_kernel, kernel_size);
    vertical_pass_kernel<<<gridSize, blockSize, 0, stream>>>(d_temp, d_dst, rows, cols,
                                                              d_kernel, kernel_size);

    // Async copy result back
    cudaMemcpyAsync(h_dst_pinned, d_dst, img_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Create result matrix from pinned memory
    cv::Mat result(rows, cols, CV_8UC1, h_dst_pinned);
    cv::Mat result_copy = result.clone();

    // Cleanup
    cudaFreeHost(h_src_pinned);
    cudaFreeHost(h_dst_pinned);
    cudaFree(d_src);
    cudaFree(d_temp);
    cudaFree(d_dst);
    cudaFree(d_kernel);
    cudaStreamDestroy(stream);

    return result_copy;
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
    gaussian_gpu_optimized(src, 15, 2.0);

    // Benchmark
    const int RUNS = 5;
    double total_ms = 0.0;
    cv::Mat result;

    for (int i = 0; i < RUNS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        result = gaussian_gpu_optimized(src, 15, 2.0);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        total_ms += duration.count();
    }

    double avg_ms = total_ms / RUNS;
    std::cout << "GPU Optimized Gaussian (avg " << RUNS << " runs): " 
              << avg_ms << " ms" << std::endl;

    cv::imwrite("output_gpu_optimized.png", result);
    std::cout << "Saved: output_gpu_optimized.png" << std::endl;

    return 0;
}