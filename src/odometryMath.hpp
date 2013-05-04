#ifndef ODOMETRY_MATH
#define ODOMETRY_MATH
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

struct Observation {
    int cameraindex;
    Eigen::Vector2d left;
    Eigen::Vector2d right;
    Eigen::Vector3d world;
};

struct Camera {
    Eigen::Matrix3d rotation;
    Eigen::Vector3d position;
};

struct BundleAdjustmentArgs {
    int num_cameras;
    int total_points;
    Eigen::Matrix<double, 3, 4> project_left;
    Eigen::Matrix<double, 3, 4> project_right;
    std::vector<Observation> observations;
    std::vector<Camera> cameras;
};

std::vector<Camera> bundleAdjust(BundleAdjustmentArgs args);

Eigen::Vector3d triangulatePoint(Eigen::Matrix<double, 3, 4> Pl,Eigen::Matrix<double, 3, 4> Pr,Eigen::Vector2d left_point,Eigen::Vector2d right_point);
std::pair<Eigen::Matrix3d,Eigen::Vector3d> computeOrientation(std::vector<Eigen::Vector3d> pts1, std::vector<Eigen::Vector3d> pts2);
std::pair<Eigen::Matrix3d,Eigen::Vector3d> computeOrientationRansac(std::vector<Eigen::Vector3d> pts1, std::vector<Eigen::Vector3d> pts2);

#endif
