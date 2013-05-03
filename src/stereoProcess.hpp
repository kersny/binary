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

        std::vector<Eigen::Vector3d> modelPoints;
        std::vector< std::pair<int, int> > modelEdges;

        StereoProcess();
        void im_pair_callback(const sm::ImageConstPtr&, const sm::ImageConstPtr&);


    private:
        Eigen::Matrix<double, 3, 1> position;
        Eigen::Vector3d worldPos;
        Eigen::Matrix3d worldRot;
        Eigen::Matrix3d orientation;
        Eigen::Vector3d modelOrigin;

        cv::Ptr<cv::FeatureDetector> detector;
        cv::Ptr<cv::DescriptorExtractor> extractor;

        std::vector<cv::KeyPoint> PL_kps, PR_kps;
        cv::Mat PL_features, PR_features;
        cv::Mat PL_mat, PR_mat;

        std::vector<cv::KeyPoint> get_keypoints(cv::Mat);
        cv::Mat extract_features(cv::Mat, std::vector<cv::KeyPoint>);
        cv::Mat im_to_opencv(const sm::ImageConstPtr&);
        void process_im_pair(const cv::Mat&, const cv::Mat&, ros::Time);

        std::vector<int> get_query_idxs(std::vector<cv::DMatch>);
        int find_kp(std::vector<int> q_idxs, int x);

        std::vector< std::vector<cv::KeyPoint> >
            get_circular_matches(std::vector< std::vector<cv::KeyPoint> >,
                                 std::vector< cv::Mat> all_fts);
};
#endif
