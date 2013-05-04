#ifndef FRAME
#define FRAME

#include <Eigen/Dense>

class Frame {
    public:
        std::vector<cv::KeyPoint> L_kps, R_kps;
        cv::Mat L_features, R_features;
        Eigen::Matrix3d Orientation_world;
        Eigen::Vector3d Translation_world;
    private:
};
#endif
