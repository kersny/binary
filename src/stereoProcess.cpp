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

#include "stereoProcess.hpp"

#define DRK cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS

std::string L_channel = "/stereo/left/image_raw";
std::string R_channel = "/stereo/right/image_raw";
uint max_im_pairs = 20;

void process_im_pair(const cv::Mat& L_mat, const cv::Mat& R_mat, ros::Time t)
{
	std::ostringstream os;
	os << "Processing image pair with timestamp: " << t << std::endl;
	debug_print(os.str(), 3);

    // Detect SIFT keypoints in both images
    debug_print("Detecting SIFT keypoints.\n", 3);
	cv::SiftDescriptorExtractor detector;
    std::vector<cv::KeyPoint> L_kps, R_kps;
    detector.detect(L_mat, L_kps);
    detector.detect(R_mat, R_kps);


    // Extract SIFT features
    debug_print("Extracting SIFT features.\n", 3);
    cv::SiftDescriptorExtractor extractor;
    cv::Mat L_features, R_features;
    extractor.compute( L_mat, L_kps, L_features );
    extractor.compute( R_mat, R_kps, R_features );

    // Find feature matches between two images
    debug_print("Matching SIFT features.\n", 3);
    cv::FlannBasedMatcher matcher;
    std::vector< cv::DMatch > matches;
    matcher.match(L_features, R_features, matches );

    if(matches.size() == 0) {
        debug_print("No matches found!\n", 2);
    } else {
        debug_print("Filtering for best matches.\n", 3);
        // Compute mean match distance
        double dist_sum = 0.0;
        for(uint i=0; i < matches.size(); i++) {
            dist_sum += matches[i].distance;
        }
        double dist_mean = dist_sum / matches.size();
        // Find standard deviation
        double sum_sq_diff = 0.0;
        for(uint i=0; i < matches.size(); i++) {
            double cur_diff = matches[i].distance - dist_mean;
            sum_sq_diff += cur_diff * cur_diff;
        }
        double std_dev = sqrt(sum_sq_diff / matches.size());
        // Refine matches by throwing out outliers
        // outlier_factor = number of standard deviations 
        //                  above mean to consider an outlier
        double outlier_factor = -0.4;
        cv::vector<cv::DMatch> good_matches;
        for(uint i=0; i < L_kps.size(); i++) {
            if(matches[i].distance < dist_mean + outlier_factor * std_dev) {
                good_matches.push_back(matches[i]);
            }
        }

        // Display matches
        cv::Mat img_matches;
        cv::drawMatches(L_mat, L_kps, R_mat, R_kps,
                        good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        cv::Mat matches_small;
        matches_small = cv::Mat::zeros(img_matches.rows / 3, img_matches.cols / 3, 16);
        cv::resize(img_matches, matches_small, matches_small.size());
        cv::namedWindow("Matches", CV_WINDOW_AUTOSIZE); 
        cv::imshow("Matches" , matches_small);
    }

    cv::Mat L_out, R_out;
    cv::drawKeypoints(L_mat, L_kps, L_out, cv::Scalar(255, 0, 0), DRK);
    cv::drawKeypoints(R_mat, R_kps, R_out, cv::Scalar(255, 0, 0), DRK);
    cv::Mat L_small, R_small;
    L_small = cv::Mat::zeros(L_out.rows / 4, L_out.cols / 4, CV_8UC1);
    R_small = cv::Mat::zeros(R_out.rows / 4, R_out.cols / 4, CV_8UC1);
    cv::resize(L_out, L_small, L_small.size());
    cv::resize(R_out, R_small, R_small.size());
    cv::namedWindow("LEFT",  CV_WINDOW_AUTOSIZE); 
    cv::namedWindow("RIGHT", CV_WINDOW_AUTOSIZE); 
    cv::imshow("LEFT" , L_small);
    cv::imshow("RIGHT", R_small);
    cv::waitKey(10);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "BINary");
    ros::NodeHandle nh;

    mf::Subscriber<sm::Image> L_sub(nh, L_channel, 1);
    mf::Subscriber<sm::Image> R_sub(nh, R_channel, 1);

    typedef mf::sync_policies::ApproximateTime<sm::Image, sm::Image> MySyncPolicy;
    mf::Synchronizer<MySyncPolicy> sync( \
        MySyncPolicy(max_im_pairs), L_sub, R_sub);
    sync.registerCallback(boost::bind(&im_pair_callback, _1, _2));

    ros::spin();

    return 0;
}