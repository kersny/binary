#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>

#include "stereoSubscriber.hpp"

cv::Mat im_to_opencv(const sm::ImageConstPtr& img)
{
    debug_print("Converting ROS sensor_msgs image to OpenCV Mat \n", 3);
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(img, sm::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        exit(0);
    }
    cv::Mat im_mat = cv_ptr->image;
    return im_mat;
}

void im_pair_callback(const sm::ImageConstPtr& L_Image,
                      const sm::ImageConstPtr& R_Image)
{
    debug_print("Recieved message pair \n", 3);
    cv::Mat L_mat = im_to_opencv(L_Image);
    cv::Mat R_mat = im_to_opencv(R_Image);

    process_im_pair(L_mat, R_mat, (*L_Image).header.stamp);
}