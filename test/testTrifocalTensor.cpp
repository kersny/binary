// Bring in my package's API, which is what I'm testing
#include "../src/trifocalTensor.hpp"
// Bring in gtest
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
float eps = 0.000000001;

#define ASSERT_TRUE_ABS_ZERO(place) ASSERT_TRUE(abs(place) < eps)

// Declare a test
TEST(trifocalTensor, testBasisProjection)
{
    Matrix<double, 3, 4> pts = Matrix<double, 3, 4>::Random();
    pts = pts * 2048;
    pts(2,0) = pts(2,1) = pts(2,2) = pts(2,3) = 1.0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            if (pts(j,i) < 0) {
                pts(j,i) = -pts(j,i);
            }
        }
    }
    Matrix<double, 3, 3> projection = computeBasisProjection(pts);
    Matrix<double, 3, 4> test = projection.inverse()*pts;
    ASSERT_TRUE_ABS_ZERO(test(0,1));
    ASSERT_TRUE_ABS_ZERO(test(0,2));
    ASSERT_TRUE_ABS_ZERO(test(1,0));
    ASSERT_TRUE_ABS_ZERO(test(1,2));
    ASSERT_TRUE_ABS_ZERO(test(2,0));
    ASSERT_TRUE_ABS_ZERO(test(2,1));
    ASSERT_TRUE_ABS_ZERO(test(0,3) - 1);
    ASSERT_TRUE_ABS_ZERO(test(1,3) - 1);
    ASSERT_TRUE_ABS_ZERO(test(2,3) - 1);
    ASSERT_TRUE(abs(test(0,0)) > eps);
    ASSERT_TRUE(abs(test(1,1)) > eps);
    ASSERT_TRUE(abs(test(2,2)) > eps);
}

TEST(trifocalTensor, testLinearFundamentalCoefficients)
{
    Matrix3d F1;
    F1 << 0, 1, 2, 3, 0, 4, 5, -(1+2+3+4+5),0;
    Matrix3d F2;
    F2 << 0, 6, 7, 8, 0, 9, 10, -(6+7+8+9+10),0;
    vector<double> roots = computeLinearFundamentalCoefficients(F1, F2);
    ASSERT_TRUE(roots.size() == 1);
    ASSERT_TRUE_ABS_ZERO(roots[0]-1.7);
}

TEST(trifocalTensor, testAreCollinear)
{
    std::vector<cv::Point2f> three_pts;
    cv::Point2f a( 1.0,  1.0);
    cv::Point2f b(10.0, 10.0);
    cv::Point2f c(15.0, 15.0);
    cv::Point2f d( 5.0, -5.0);
    three_pts.push_back(a); 
    three_pts.push_back(b);
    three_pts.push_back(c);
    ASSERT_TRUE(are_collinear(three_pts,  1.0)); // A B C are definitely collinear
    ASSERT_TRUE(are_collinear(three_pts,  5.0)); 
    ASSERT_TRUE(are_collinear(three_pts, 10.0)); 
    three_pts.pop_back();
    three_pts.push_back(d);
    // A B D are not collinear with <= 5 threshold
    ASSERT_FALSE(are_collinear(three_pts, 1.0)); 
    ASSERT_FALSE(are_collinear(three_pts, 5.0));
    // A B D are collinear with threshold >= 10
    ASSERT_TRUE(are_collinear(three_pts, 50.0));
    cv::Point2f x(-5.0,  0.0);
    cv::Point2f y( 5.0,  0.0);
    cv::Point2f z( 0.0, 10.0);
    three_pts.clear();
    three_pts.push_back(x);
    three_pts.push_back(y);
    three_pts.push_back(z);
    // X Y Z are collinear with threshold >= 10 and >= 8.95
    ASSERT_TRUE(are_collinear(three_pts, 10.0));
    ASSERT_TRUE(are_collinear(three_pts, 8.95));
    ASSERT_FALSE(are_collinear(three_pts, 8.90));
    ASSERT_FALSE(are_collinear(three_pts, 5.00));
    ASSERT_FALSE(are_collinear(three_pts, 0.00));
}

TEST(trifocalTensor, testAreSome3Collinear)
{
    std::vector<cv::Point2f> four_pts;
    cv::Point2f a( 1.0,  1.0);
    cv::Point2f b(10.0, 10.0);
    cv::Point2f c(15.0, 15.0);
    cv::Point2f d( 5.0, -5.0);
    four_pts.push_back(a); 
    four_pts.push_back(b);
    four_pts.push_back(c);
    four_pts.push_back(d);
    // A B C are exactly collinear so this is true for any positive threshold
    ASSERT_TRUE(are_some_3_collinear(four_pts,  0.1));
    ASSERT_TRUE(are_some_3_collinear(four_pts,  1.0));
    ASSERT_TRUE(are_some_3_collinear(four_pts, 10.0));
    ASSERT_TRUE(are_some_3_collinear(four_pts)); // test default threshold
    cv::Point2f x(20.0, 20.0); // Rectangle
    cv::Point2f y(30.0, 20.0);
    cv::Point2f z(20.0, 30.0);
    cv::Point2f w(30.0, 30.0);
    four_pts.clear();
    four_pts.push_back(x); 
    four_pts.push_back(y);
    four_pts.push_back(z);
    four_pts.push_back(w);
    // For any threshold < 7.071 (distance from center of rect to point),
    //  no 3 points can be collinear
    ASSERT_FALSE(are_some_3_collinear(four_pts,  0.1));
    ASSERT_FALSE(are_some_3_collinear(four_pts,  1.0));
    ASSERT_FALSE(are_some_3_collinear(four_pts, 7.06));
    ASSERT_TRUE(are_some_3_collinear(four_pts,  7.08));
    ASSERT_TRUE(are_some_3_collinear(four_pts,  50.0));
}


// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
