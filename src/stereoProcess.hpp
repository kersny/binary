#ifndef STEREO_PROCESS
#define STEREO_PROCESS

#include <ros/ros.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>

#include <Eigen/Dense>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <string>
#include <assert.h>

#include "omp.h"

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
        std::vector<cv::KeyPoint> P_kps;
        cv::Mat P_features;
        cv::Mat P_mat;

        std::vector<cv::KeyPoint> get_keypoints(cv::Mat);
        cv::Mat extract_features(cv::Mat, std::vector<cv::KeyPoint>);
        cv::Mat im_to_opencv(const sm::ImageConstPtr&);
        void process_im_pair(const cv::Mat&, const cv::Mat&, ros::Time);
};

class TripleMatches {
    public:
        // weighting for sum of keypoints responses
        static const double kp_weight = 1.0;
        // note in opencv < 2.4.4 keypoint responses will all be 0
        static const double match_dist_weight = 1.0;

        std::vector<cv::KeyPoint> L_kps;
        std::vector<cv::KeyPoint> R_kps;
        std::vector<cv::KeyPoint> P_kps;
        std::vector<double> weights;
            // weight of the triple match is a function of
            //  keypoint responses and match distances
};

#endif
