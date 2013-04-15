#ifndef STEREO_PROCESS
#define STEREO_PROCESS

#include <ros/ros.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <string>

#include "utilities.hpp"

#define sm sensor_msgs
#define mf message_filters

class StereoProcess {
    public:
        std::string L_channel;
        std::string R_channel;
        uint max_im_pairs;

        StereoProcess();
        void im_pair_callback(const sm::ImageConstPtr&, const sm::ImageConstPtr&);

    private:
        std::vector<cv::KeyPoint> get_keypoints(cv::Mat);
        cv::Mat extract_features(cv::Mat, std::vector<cv::KeyPoint>);
        cv::Mat im_to_opencv(const sm::ImageConstPtr&);
        void process_im_pair(const cv::Mat&, const cv::Mat&, ros::Time);
};

#endif