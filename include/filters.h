// include/filters.h
#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>

cv::Mat gaussian_naive(const cv::Mat& src, int kernel_size, double sigma);

#endif