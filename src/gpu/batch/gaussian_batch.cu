// Phase 2.3: gaussian_batch.cu
// GPU batch processing - queue multiple images to amortize PCIe overhead

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <vector>

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

// Process a batch of images
std::vector<cv::Mat> gaussian_gpu_batch(const std::vector<cv::Mat>& batch,
                                        int kernel_size, double sigma) {
    if (batch.empty()) return {};

    int num_images = batch.size();
    int rows = batch[0].rows;
    int cols = batch[0].cols;
    size_t img_bytes = rows * cols;

    std::vector<float> h_kernel = build_kernel(kernel_size, sigma);

    // Allocate pinned memory for entire batch
    uchar* h_batch_src = nullptr;
    uchar* h_batch_dst = nullptr;
    cudaMallocHost(&h_batch_src, img_bytes * num_images);
    cudaMallocHost(&h_batch_dst, img_bytes * num_images);

    // Copy batch to pinned memory
    for (int i = 0; i < num_images; i++) {
        memcpy(h_batch_src + i * img_bytes, batch[i].data, img_bytes);
    }

    // Device memory for batch
    uchar* d_batch_src = nullptr;
    float* d_batch_temp = nullptr;
    uchar* d_batch_dst = nullptr;
    float* d_kernel = nullptr;

    cudaMalloc(&d_batch_src, img_bytes * num_images);
    cudaMalloc(&d_batch_temp, img_bytes * num_images * sizeof(float));
    cudaMalloc(&d_batch_dst, img_bytes * num_images);
    cudaMalloc(&d_kernel, h_kernel.size() * sizeof(float));

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Upload batch
    cudaMemcpyAsync(d_batch_src, h_batch_src, img_bytes * num_images,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_kernel, h_kernel.data(), h_kernel.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // Process each image in batch
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);

    for (int i = 0; i < num_images; i++) {
        uchar* src_offset = d_batch_src + i * img_bytes;
        float* temp_offset = d_batch_temp + i * img_bytes;
        uchar* dst_offset = d_batch_dst + i * img_bytes;

        horizontal_pass_kernel<<<gridSize, blockSize, 0, stream>>>(
            src_offset, (float*)temp_offset, rows, cols, d_kernel, kernel_size);

        vertical_pass_kernel<<<gridSize, blockSize, 0, stream>>>(
            (float*)temp_offset, dst_offset, rows, cols, d_kernel, kernel_size);
    }

    // Download batch
    cudaMemcpyAsync(h_batch_dst, d_batch_dst, img_bytes * num_images,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Create result vector
    std::vector<cv::Mat> results;
    for (int i = 0; i < num_images; i++) {
        cv::Mat result(rows, cols, CV_8UC1, h_batch_dst + i * img_bytes);
        results.push_back(result.clone());
    }

    // Cleanup
    cudaFreeHost(h_batch_src);
    cudaFreeHost(h_batch_dst);
    cudaFree(d_batch_src);
    cudaFree(d_batch_temp);
    cudaFree(d_batch_dst);
    cudaFree(d_kernel);
    cudaStreamDestroy(stream);

    return results;
}

int main(int argc, char* argv[]) {
    std::string image_path = (argc > 1) ? argv[1] : "test_image.png";

    cv::Mat src = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << src.cols << "x" << src.rows << std::endl;

    // Create batch of 100 copies (simulating 100 images)
    std::vector<cv::Mat> batch(100, src);
    std::cout << "Batch size: " << batch.size() << " images" << std::endl;

    // Warm up
    gaussian_gpu_batch(batch, 15, 2.0);

    // Benchmark batch processing
    const int RUNS = 3;
    double total_ms = 0.0;
    std::vector<cv::Mat> results;

    for (int i = 0; i < RUNS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        results = gaussian_gpu_batch(batch, 15, 2.0);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        total_ms += duration.count();
    }

    double avg_ms = total_ms / RUNS;
    double avg_per_image = avg_ms / batch.size();
    double throughput = 1000.0 / avg_per_image;  // images per second

    std::cout << "GPU Batch Gaussian (avg " << RUNS << " runs):" << std::endl;
    std::cout << "  Total time: " << avg_ms << " ms" << std::endl;
    std::cout << "  Per image: " << avg_per_image << " ms" << std::endl;
    std::cout << "  Throughput: " << throughput << " img/sec" << std::endl;

    // Save first result
    cv::imwrite("output_gpu_batch_0.png", results[0]);
    std::cout << "Saved: output_gpu_batch_0.png" << std::endl;

    return 0;
}