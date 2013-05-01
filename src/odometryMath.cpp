#include "odometryMath.hpp"
#include <opencv2/core/eigen.hpp>

Eigen::Vector3d triangulatePoint(cv::Mat Pl,cv::Mat Pr,cv::Point2f left_point,cv::Point2f right_point)
{
    Eigen::Matrix<double,3,4> Ple,Pre;
    Eigen::Matrix<double,3,1> lp,rp;
    lp(0,0) = (double)left_point.x;
    lp(1,0) = (double)left_point.y;
    lp(2,0) = 1.0;
    rp(0,0) = (double)right_point.x;
    rp(1,0) = (double)right_point.y;
    rp(2,0) = 1.0;
    Eigen::Matrix<double,6,4> A;
    Eigen::Matrix<double,6,1> b;
    cv::cv2eigen(Pl,Ple);cv::cv2eigen(Pr,Pre);
    A.block<3,4>(0,0) = Ple;
    A.block<3,4>(3,0) = Pre;
    b.block<3,1>(0,0) = lp;
    b.block<3,1>(3,0) = rp;
    Eigen::Vector4d result = (A.transpose()*A).inverse()*(A.transpose()*b);
    return result.block<3,1>(0,0) / result(3,0);
}

std::pair<Eigen::Matrix3d,Eigen::Vector3d> computeOrientation(std::vector<Eigen::Vector3d> pts1, std::vector<Eigen::Vector3d> pts2)
{
    Eigen::Vector3d now_avg,prev_avg;
    now_avg << 0.0,0.0,0.0;
    prev_avg << 0.0,0.0,0.0;
    for (unsigned int i = 0; i < pts1.size(); i++) {
        now_avg += pts1[i];
        prev_avg += pts2[i];
    }
    Eigen::Vector3d centroid_now = now_avg/pts1.size();
    Eigen::Vector3d centroid_prev = prev_avg/pts1.size();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (unsigned int i = 0; i < pts1.size(); i++) {
        cov += (pts1[i] - centroid_now)*((pts2[i] - centroid_prev).transpose());
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> cov_svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R = cov_svd.matrixV()*(cov_svd.matrixU().transpose());
    //Eigen::Matrix3d R = cov_svd.matrixU()*(cov_svd.matrixV().transpose());
    if (R.determinant() < 0) {
        R.col(2) = -1*R.col(2);
    }
    Eigen::Vector3d trans = -R*centroid_now + centroid_prev;
    return std::make_pair(R,trans);
}
std::pair<Eigen::Matrix3d,Eigen::Vector3d> computeOrientationRansac(std::vector<Eigen::Vector3d> points_curr, std::vector<Eigen::Vector3d> points_prev)
{
    double maxRatio = 0;
    int iter = 0;
    std::vector<Eigen::Vector3d> pts1_all, pts2_all; // all inliers
    while (iter < 250) {
        std::vector<int> indices;
        for (unsigned int i = 0; i < points_curr.size(); i++) {
            indices.push_back(i);
        }
        std::random_shuffle(indices.begin(), indices.end());
        std::vector<Eigen::Vector3d> pts1,pts2;
        for (unsigned int i = 0; i < 3; i++) {
            pts1.push_back(points_curr[indices[i]]);
            pts2.push_back(points_prev[indices[i]]);
        }

        std::pair<Eigen::Matrix3d,Eigen::Vector3d> ans = computeOrientation(pts1, pts2);

        int inlierCount = 0;
        std::vector<int> inliers;
        for (unsigned int i = 0; i < points_curr.size(); i++) {
            if (((ans.first*(points_curr[i]) + ans.second) - points_prev[i]).norm() < 10.0) {
                inliers.push_back(i);
                inlierCount++;
            }
        }
        double inlierRatio = ((double)inlierCount)/((double)points_curr.size());
        if (inlierRatio > maxRatio) {
            pts1_all.clear();
            pts2_all.clear();
            for (unsigned int i = 0; i < inliers.size(); i++) {
                pts1_all.push_back(points_curr[inliers[i]]);
                pts2_all.push_back(points_prev[inliers[i]]);
            }
            maxRatio = inlierRatio;
        }
        iter++;
    }
    return computeOrientation(pts1_all,pts2_all);
}
