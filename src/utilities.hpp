#ifndef BINARY_UTILITIES
#define BINARY_UTILITIES

#include <iostream>
#include <string>
#include <iomanip>

#include <opencv2/opencv.hpp>

#define uint unsigned int

void debug_print(std::string, uint);
cv::Mat norm_by_index(cv::Mat, int, int);
std::string ppmd(cv::Mat);

#endif