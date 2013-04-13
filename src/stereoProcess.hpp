#ifndef STEREO_PROCESS
#define STEREO_PROCESS

#include "utilities.hpp"
#include "stereoSubscriber.hpp"

#define sm sensor_msgs
#define mf message_filters

void process_im_pair(const cv::Mat&, const cv::Mat&, ros::Time);

#endif