#include "odometryMath.hpp"
#include <iomanip>
#include <opencv2/core/eigen.hpp>

Eigen::Vector3d triangulatePoint_linear_me(Eigen::Matrix<double, 3, 4> Pl,Eigen::Matrix<double, 3, 4> Pr,Eigen::Vector2d left_point,Eigen::Vector2d right_point)
{
    Eigen::Matrix<double,6,4> A;
    Eigen::Matrix<double,6,1> b;
    A.block<3,4>(0,0) = Pl;
    A.block<3,4>(3,0) = Pr;
    b.block<2,1>(0,0) = left_point;
    b(2,0) = 1.0;
    b.block<2,1>(3,0) = right_point;
    b(5,0) = 1.0;
    //Eigen::Vector4d result = (A.transpose()*A).inverse()*(A.transpose()*b);
    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 4> > pt_svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector4d result = pt_svd.solve(b);
    return result.block<3,1>(0,0) / result(3,0);
}
Eigen::Vector3d triangulatePoint_linear_ls(Eigen::Matrix<double, 3, 4> Pl,Eigen::Matrix<double, 3, 4> Pr,Eigen::Vector2d left_point,Eigen::Vector2d right_point)
{
    Eigen::Matrix<double,4, 3> A;
    Eigen::Matrix<double, 4, 1> b;
    A << left_point(0)*Pl(2,0)-Pl(0,0),left_point(0)*Pl(2,1)-Pl(0,1),left_point(0)*Pl(2,2)-Pl(0,2),
         left_point(1)*Pl(2,0)-Pl(1,0),left_point(1)*Pl(2,1)-Pl(0,1),left_point(1)*Pl(2,2)-Pl(1,2),
         right_point(0)*Pr(2,0)-Pr(0,0),right_point(0)*Pr(2,1)-Pr(0,1),right_point(0)*Pr(2,2)-Pr(0,2),
         right_point(1)*Pr(2,0)-Pr(1,0),right_point(1)*Pr(2,1)-Pr(0,1),right_point(1)*Pr(2,2)-Pr(1,2);
    b << -(left_point(0)*Pl(2,3) - Pl(0,3)), -(left_point(1)*Pl(2,3) - Pl(1,3)),
         -(right_point(0)*Pr(2,3) - Pr(0,3)), -(right_point(1)*Pr(2,3) - Pr(1,3));
    Eigen::JacobiSVD<Eigen::Matrix<double, 4, 3> > pt_svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return pt_svd.solve(b);
}
Eigen::Vector3d triangulatePoint_linear_eigen(Eigen::Matrix<double, 3, 4> Pl,Eigen::Matrix<double, 3, 4> Pr,Eigen::Vector2d left_point,Eigen::Vector2d right_point, double wl = 1.0, double wr = 1.0)
{
    Eigen::Matrix<double, 4, 4> A;
    A << wl*left_point(0)*Pl.row(2) - Pl.row(0), wl*left_point(1)*Pl.row(2) - Pl.row(1),
         wr*right_point(0)*Pr.row(2) - Pr.row(0), wr*right_point(1)*Pr.row(2) - Pr.row(1);
    Eigen::JacobiSVD<Eigen::Matrix<double, 4, 4> > pt_svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector4d ans = pt_svd.matrixV().col(3);
    return ans.block<3,1>(0,0)/ans(3,0);
}

Eigen::Vector3d triangulatePoint_nonlinear_eigen(Eigen::Matrix<double, 3, 4> Pl,Eigen::Matrix<double, 3, 4> Pr,Eigen::Vector2d left_point,Eigen::Vector2d right_point)
{
    Eigen::Vector3d estimate = triangulatePoint_linear_eigen(Pl,Pr,left_point,right_point);
    std::cout << std::setprecision(15);
    std::cout << "Initial: " << estimate << std::endl;
    Eigen::Matrix<double, 4, 1> estimate_homog;
    estimate_homog << estimate, 1.0;
    double weight_left = 1.0/(Pl.row(2).dot(estimate_homog));
    double weight_right = 1.0/(Pr.row(2).dot(estimate_homog));
    double delta_weight_left = INFINITY;
    double delta_weight_right = INFINITY;
    int iters = 0;
    while (delta_weight_left > 0.0001 && delta_weight_right > 0.0001 && iters < 10) {
        estimate = triangulatePoint_linear_eigen(Pl,Pr,left_point,right_point,weight_left,weight_right);
        estimate_homog(0) = estimate(0);
        estimate_homog(1) = estimate(1);
        estimate_homog(2) = estimate(2);
        estimate_homog(3) = 1.0;
        double new_weight_left = 1.0/(Pl.row(2).dot(estimate_homog));
        double new_weight_right = 1.0/(Pr.row(2).dot(estimate_homog));
        delta_weight_left = abs(new_weight_left - weight_left);
        delta_weight_right = abs(new_weight_right - weight_right);
        weight_left = new_weight_left;
        weight_right = new_weight_right;
        iters++;
    }
    std::cout << "Final: " << estimate << std::endl;
    std::cout << "Nonlinear triangulation iterations:" << iters << std::endl;
    return estimate;
}
Eigen::Vector3d triangulatePoint_cv(cv::Mat Pl,cv::Mat Pr,cv::Point2f left_point,cv::Point2f right_point)
{
    cv::Mat outp;
    cv::Mat left_points_m(2,1,CV_64F);
    cv::Mat right_points_m(2,1,CV_64F);
    left_points_m.at<double>(0,0) = left_point.x;
    left_points_m.at<double>(1,0) = left_point.y;
    right_points_m.at<double>(0,0) = right_point.x;
    right_points_m.at<double>(1,0) = right_point.y;
    cv::triangulatePoints(Pl,Pr,left_points_m,right_points_m,outp);
    Eigen::Vector3d ret;
    ret << outp.at<double>(0,0)/outp.at<double>(3,0),
           outp.at<double>(1,0)/outp.at<double>(3,0),
           outp.at<double>(2,0)/outp.at<double>(3,0);
    return ret;
}
Eigen::Vector3d triangulatePoint(cv::Mat Pl,cv::Mat Pr,cv::Point2f left_point,cv::Point2f right_point)
{
    Eigen::Matrix<double,3,4> Ple,Pre;
    Eigen::Matrix<double,2,1> lp,rp;
    lp(0,0) = (double)left_point.x;
    lp(1,0) = (double)left_point.y;
    rp(0,0) = (double)right_point.x;
    rp(1,0) = (double)right_point.y;
    cv::cv2eigen(Pl,Ple);cv::cv2eigen(Pr,Pre);
    return triangulatePoint_linear_eigen(Ple,Pre,lp,rp);
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
