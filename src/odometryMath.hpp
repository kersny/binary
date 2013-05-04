#ifndef ODOMETRY_MATH
#define ODOMETRY_MATH
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

Eigen::Vector3d triangulatePoint(Eigen::Matrix<double, 3, 4> Pl,Eigen::Matrix<double, 3, 4> Pr,Eigen::Vector2d left_point,Eigen::Vector2d right_point);
std::pair<Eigen::Matrix3d,Eigen::Vector3d> computeOrientation(std::vector<Eigen::Vector3d> pts1, std::vector<Eigen::Vector3d> pts2);
std::pair<Eigen::Matrix3d,Eigen::Vector3d> computeOrientationRansac(std::vector<Eigen::Vector3d> pts1, std::vector<Eigen::Vector3d> pts2);

#endif
