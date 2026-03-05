// include/utils.h
#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>

// Timing utilities
class Timer {
public:
    Timer();
    void start();
    void stop();
    double elapsed_ms() const;  
    
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
};


cv::Mat load_image(const std::string& path);
void save_image(const std::string& path, const cv::Mat& img);
void display_image(const std::string& window_name, const cv::Mat& img);

#endif