// Bring in my package's API, which is what I'm testing
#include "../src/odometryMath.hpp"
// Bring in gtest
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
float eps = 0.000000001;

#define ASSERT_TRUE_ABS_ZERO(place) ASSERT_TRUE(abs(place) < eps)

const int im_w = 1450;
const int im_h = 1950;

// Declare a test
TEST(math, testRandomTriangulation)
{
    cv::Mat Pl = (cv::Mat_<double>(3,4) << 1107.58877335145, 0, 703.563442850518, 0, 0, 1105.93566117489, 963.193789785819, 0, 0, 0, 1, 0);
    cv::Mat Pr = (cv::Mat_<double>(3,4) << 1105.57021914223,6.18934957543074,759.754258185686,-612760.0875376,9.71869909913803, 1123.12983099782,941.444195743573,-1240.37638207625, 0.001724650725893,0.0187425606411105,0.999822855310123,-0.789271765090486);
    Eigen::Matrix<double, 3, 4> Pl_E,Pr_E;
    cv::cv2eigen(Pl,Pl_E);cv::cv2eigen(Pr,Pr_E);
    int tests = 0;
    while(tests < 100) {
        Eigen::Vector3d point = 8000*Eigen::Vector3d::Random();
        if (point(2) < 0) {
            point(2) = -1*point(2);
        }
        Eigen::Vector4d homo_point;
        homo_point << point(0),point(1),point(2), 1.0;
        Eigen::Vector3d homo_left = Pl_E*homo_point;
        Eigen::Vector3d homo_right = Pr_E*homo_point;
        homo_left = homo_left/homo_left(2);
        homo_right = homo_right/homo_right(2);
        if(homo_left(0) < 0 || homo_left(1) < 0 || 
           homo_right(0) < 0 || homo_right(1) < 0 ||
           homo_left(0) > im_w || homo_left(1) > im_h || 
           homo_right(0) > im_w || homo_right(1) > im_h ) {
            // this test isn't good, points aren't actually in images
            continue;
        } else {
            tests++;
        }
        //std::cout << "\n\nHomogenous World Pt: \n" << homo_point << std::endl;
        //std::cout << " L: \n" << homo_left << " \n R: \n" << homo_right << std::endl;
        cv::Point2f leftp,rightp;
        leftp.x = homo_left(0);
        leftp.y = homo_left(1);
        rightp.x = homo_right(0);
        rightp.y = homo_right(1);
        Eigen::Vector3d pt = triangulatePoint(Pl,Pr,leftp,rightp);
        //std::cout << "\n Dist: \n" << (pt - point) << std::endl;
        ASSERT_TRUE_ABS_ZERO((pt - point).norm());
    }
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
