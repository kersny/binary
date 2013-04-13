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
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <string>

#include "stereoProcess.hpp"

std::string L_channel = "/stereo/left/image_raw";
std::string R_channel = "/stereo/right/image_raw";
uint max_im_pairs = 10;

void process_im_pair(const cv::Mat& L_mat, const cv::Mat& R_mat, ros::Time t)
{
	std::ostringstream os;
	os << "Processing image pair with timestamp: " << t << std::endl;
	debug_print(os.str(), 3);

	cv::SiftDescriptorExtractor detector;
    std::vector<cv::KeyPoint> L_kps, R_kps;
    detector.detect(L_mat, L_kps);
    detector.detect(R_mat, R_kps);

    cv::Mat L_out, R_out;
    cv::drawKeypoints(L_mat, L_kps, L_out);
    cv::drawKeypoints(R_mat, R_kps, R_out);
    cv::namedWindow("LEFT",  CV_WINDOW_NORMAL); 
    cv::namedWindow("RIGHT", CV_WINDOW_NORMAL); 
    cv::imshow("LEFT" , L_out);     
    cv::imshow("RIGHT", R_out);     
    cv::waitKey(0);
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