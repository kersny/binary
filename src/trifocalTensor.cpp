#include "trifocalTensor.hpp"
#include <iostream>
#include <gsl/gsl_poly.h>
#include <assert.h>

#define D(i,j,k) (dets[(i-1)*4+(j-1)*2+(k-1)])

/*
 * Computes H (3x3) such that pts = H*[eye(3) [1 1 1]]
 */

Matrix3d computeBasisProjection(Matrix<double, 3, 4> pts) {
    Vector3d v = pts.block<3,3>(0,0).inverse() * pts.block<3,1>(0, 3);
    Matrix3d d = v.asDiagonal();
    Matrix3d proj = pts.block<3,3>(0,0)*d;
    return proj;
}


vector<double> computeLinearFundamentalCoefficients(Matrix3d F1, Matrix3d F2) {
    vector<Matrix3d> srcs;
    srcs.push_back(F1);
    srcs.push_back(F2);
    vector<double> dets(8);
    for (int i1 = 0; i1 < 2; i1++) {
	for (int i2 = 0; i2 < 2; i2++) {
	    for (int i3 = 0; i3 < 2; i3++) {
		Matrix3d target;
		target.block<3,1>(0,0) = srcs[i1].block<3,1>(0,0);
		target.block<3,1>(0,1) = srcs[i2].block<3,1>(0,1);
		target.block<3,1>(0,2) = srcs[i3].block<3,1>(0,2);
		D(i1+1,i2+1,i3+1) = target.determinant();
	    }
	}
    }
    double base = -D(2,1,1)+D(1,2,2)+D(1,1,1)+D(2,2,1)+D(2,1,2)-D(1,2,1)-D(1,1,2)-D(2,2,2);
    double a = (D(1,1,2)-2*D(1,2,2)-2*D(2,1,2)+D(2,1,1)-2*D(2,2,1)+D(1,2,1)+3*D(2,2,2))/base;
    double b = (D(2,2,1)+D(1,2,2)+D(2,1,2)-3*D(2,2,2))/base;
    double c = (D(2,2,2))/base;
    double rets[3];
    int r = gsl_poly_solve_cubic(a, b, c, &rets[0], &rets[1], &rets[2]);
    vector<double> ret;
    for (int i = 0; i < r; i++) {
	ret.push_back(rets[i]);
    }
    return ret;
}

Matrix<double,1,5> computeFPQRST(Vector3d x_one_hat, Vector3d x_two_hat) {
    Matrix<double,1,5> ret;
    ret <<
	x_one_hat(1)*x_two_hat(0) - x_one_hat(1)*x_two_hat(2),
	x_one_hat(2)*x_two_hat(0) - x_one_hat(1)*x_two_hat(2),
	x_one_hat(0)*x_two_hat(1) - x_one_hat(1)*x_two_hat(2),
	x_one_hat(2)*x_two_hat(1) - x_one_hat(1)*x_two_hat(2),
	x_one_hat(0)*x_two_hat(2) - x_one_hat(1)*x_two_hat(2);
    return ret;
}

Matrix<double, 3, 3> matFromCameraPoint(Vector3d camera_point)
{
    Matrix<double, 3, 3> ret;
    ret << 0, camera_point(2), -camera_point(1),
	   -camera_point(2), 0, camera_point(0),
	   camera_point(1), -camera_point(0), 0;
    return ret;
}
Matrix<double, 3, 4> matFromWorldPoint(Vector4d world_point)
{
    Matrix<double, 3, 4> ret;
    ret << world_point(0), 0, 0, world_point(3),
	   0, world_point(1), 0, world_point(3),
	   0, 0, world_point(2), world_point(3);
    return ret;
}

Matrix<double, 3, 4> computeProjection(Matrix<double, 4, 6> world_points, Matrix<double, 3, 6> camera_pts) {
    Matrix<double, 6, 4> A;
    Matrix<double, 3, 4> projectionBasis;
    projectionBasis << 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1;
    A.block<3,4>(0,0) = matFromCameraPoint(camera_pts.col(0))*matFromWorldPoint(world_points.col(0));
    A.block<3,4>(3,0) = matFromCameraPoint(camera_pts.col(1))*projectionBasis;
    JacobiSVD<MatrixXd> A_svd(A, ComputeFullU | ComputeFullV);
    Vector4d pvec = A_svd.matrixV().col(3);
    return matFromWorldPoint(pvec);
}

vector<vector<Matrix<double, 3, 4> > > computeTensorCandidates(vector<Matrix<double, 3, 6> > pts) {
    assert(pts.size() == 3);
    vector<vector<Matrix<double, 3, 4> > > ret;
    Matrix<double, 3, 5> foo;
    vector<Matrix<double, 3, 6> > projected_pts;
    vector<Matrix3d> projections;
    for (int i = 0; i < 3; i++) {
	Matrix3d basisProjection = computeBasisProjection(pts[i].block<3,4>(0,2));
	projections.push_back(basisProjection);
	Matrix<double, 3, 6> projected;
	for (int j = 0; j < 6; j++) {
	    projected.block<3,1>(0,j) = basisProjection.inverse()*pts[i].block<3,1>(0,j);
	}
	projected_pts.push_back(projected);
	Matrix<double, 1, 5> pqrst_vals = computeFPQRST(projected.block<3,1>(0,0), projected.block<3,1>(0,1));
	foo.block<1,5>(i,0) = pqrst_vals;
    }
    JacobiSVD<MatrixXd> svd(foo, ComputeFullU | ComputeFullV);
    Matrix<double, 5, 2> parts = svd.matrixV().block<5, 2>(0,3);
    Matrix3d F1;
    Matrix<double, 5, 1> p1 = parts.block<5,1>(0,0);
    F1 << 0, p1(0), p1(1), p1(2), 0, p1(3), p1(4), -(p1(0)+p1(1)+p1(2)+p1(3)+p1(4)), 0;
    Matrix3d F2;
    Matrix<double, 5, 1> p2 = parts.block<5,1>(0,1);
    F2 << 0, p2(0), p2(1), p2(2), 0, p2(3), p2(4), -(p2(0)+p2(1)+p2(2)+p2(3)+p2(4)), 0;
    vector<double> roots = computeLinearFundamentalCoefficients(F1, F2);
    for (uint i = 0; i < roots.size(); i++) {
	if (roots[i] < 0) continue;
	//cout << "Computing Results for root: " << roots[i] << endl;
	Matrix3d F = roots[i]*F1 + (1-roots[i])*F2;
	Matrix<double, 3, 3> Fa;
	Fa << F(0,1), F(1,0), 0, F(0,2), 0, F(2,0), 0, F(1,2), F(2,1);
	JacobiSVD<MatrixXd> svd_fa(Fa, ComputeFullU | ComputeFullV);
	Matrix<double, 3, 1> ABC = svd_fa.matrixV().col(2);
	Matrix<double, 3, 3> Fb = F.transpose();
	JacobiSVD<MatrixXd> svd_fb(Fb, ComputeFullU | ComputeFullV);
	Matrix<double, 3, 1> KLM = svd_fb.matrixV().col(2);


	Matrix<double, 6, 4> abcd;
	abcd << 0, -ABC(2), ABC(1), 0,
		ABC(2), 0, -ABC(0), 0,
		-ABC(1), ABC(0), 0, 0,
		KLM(1), -KLM(0), 0, KLM(0)-KLM(1),
		0, KLM(2), -KLM(1), KLM(1)-KLM(2),
		-KLM(2), 0, KLM(1), KLM(2)-KLM(0);
	JacobiSVD<MatrixXd> svd_abcd(abcd, ComputeFullU | ComputeFullV);
	Matrix<double, 4, 1> params = svd_abcd.matrixV().col(3);
	Matrix<double, 4, 6> world_points;
	double a,b,c,d;
	a = params(0);
	b = params(1);
	c = params(2);
	d = params(3);
	Vector4d X1,X2,X3,X4,X5,X6;
	world_points << a, 1, 1, 0, 0, 0,
			b, 1, 0, 1, 0, 0,
			c, 1, 0, 0, 1, 0,
			d, 1, 0, 0, 0, 1;

	Matrix<double, 3, 4> P1 = projections[0]*computeProjection(world_points, projected_pts[0]);
	Matrix<double, 3, 4> P2 = projections[1]*computeProjection(world_points, projected_pts[1]);
	Matrix<double, 3, 4> P3 = projections[2]*computeProjection(world_points, projected_pts[2]);
	// fix signs?
	vector<Matrix<double, 3, 4> > projs;
	projs.push_back(P1);
	projs.push_back(P2);
	projs.push_back(P3);
	ret.push_back(projs);
    }
    return ret;
}
