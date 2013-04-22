#ifndef BINARY_UTILITIES
#define BINARY_UTILITIES

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#define uint unsigned int

void debug_print(std::string, uint);

std::string ppmd(cv::Mat m);

#endif