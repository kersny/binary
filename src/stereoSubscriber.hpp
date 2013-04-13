#ifndef STEREO_SUBSCRIBER
#define STEREO_SUBSCRIBER

#define sm sensor_msgs
#define mf message_filters

#include "stereoProcess.hpp"
#include "utilities.hpp"

cv::Mat im_to_opencv(const sm::ImageConstPtr&);
void im_pair_callback(const sm::ImageConstPtr&, const sm::ImageConstPtr&);

#endif