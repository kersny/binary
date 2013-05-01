#ifndef ODOMETRY_MATH
#define ODOMETRY_MATH
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

Eigen::Vector3d triangulatePoint(cv::Mat Pl,cv::Mat Pr,cv::Point2f left_point,cv::Point2f right_point);
std::pair<Eigen::Matrix3d,Eigen::Vector3d> computeOrientation(std::vector<Eigen::Vector3d> pts1, std::vector<Eigen::Vector3d> pts2);
std::pair<Eigen::Matrix3d,Eigen::Vector3d> computeOrientationRansac(std::vector<Eigen::Vector3d> pts1, std::vector<Eigen::Vector3d> pts2);

#endif
