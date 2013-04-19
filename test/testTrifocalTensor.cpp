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


// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
testing::InitGoogleTest(&argc, argv);
return RUN_ALL_TESTS();
}
