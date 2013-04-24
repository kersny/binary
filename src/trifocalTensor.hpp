#ifndef TRIFOCAL_TENSOR
#define TRIFOCAL_TENSOR

#include <Eigen/Dense>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "stereoProcess.hpp"

using namespace std;
using namespace Eigen;

Matrix3d computeBasisProjection(Matrix<double, 3, 4> pts);
vector<double> computeLinearFundamentalCoefficients(Matrix3d F1, Matrix3d F2);
Matrix<double,1,5> computeFPQRST(Vector3d x_one_hat, Vector3d x_two_hat);
Matrix3d matFromCameraPoint(Vector3d camera_point);
Matrix<double, 3, 4> matFromWorldPoint(Vector4d world_point);
Matrix<double, 3, 4> computeProjection(Matrix<double, 4, 6> world_points, Matrix<double, 3, 6> camera_pts);
vector<Matrix<double, 3, 4> > computeTensor(TripleMatches m);
vector<vector<Matrix<double, 3, 4> > > computeTensorCandidates(vector<Matrix<double, 3, 6> > pts);

bool are_collinear(std::vector<cv::Point2f> pts, double dist_threshold);
bool are_some_3_collinear(std::vector<cv::Point2f> pts, double dist_threshold);
bool are_some_3_collinear(std::vector<cv::Point2f> pts);

#endif // TRIFOCAL_TENSOR
