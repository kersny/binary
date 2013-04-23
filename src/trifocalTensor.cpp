#include "trifocalTensor.hpp"
#include <iostream>
#include <gsl/gsl_poly.h>
#include <assert.h>
#include "utilities.hpp"
#include <algorithm>

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
	x_one_hat(0)*x_two_hat(1) - x_one_hat(2)*x_two_hat(1),
	x_one_hat(0)*x_two_hat(2) - x_one_hat(2)*x_two_hat(1),
	x_one_hat(1)*x_two_hat(0) - x_one_hat(2)*x_two_hat(1),
	x_one_hat(1)*x_two_hat(2) - x_one_hat(2)*x_two_hat(1),
	x_one_hat(2)*x_two_hat(0) - x_one_hat(2)*x_two_hat(1);
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


typedef std::pair<int,int> intpair;
bool pair_comp( const intpair& l, const intpair& r) {
    return l.second < r.second;
}

vector<Matrix<double, 3, 4> > computeTensor(TripleMatches t) {
    // Sort triple-matches by weight
    std::vector<intpair> indexed_weights;
    for(uint i = 0 ; i < t.weights.size() ; i++) {
	intpair cur_pair = std::make_pair(i,t.weights.at(i));
	indexed_weights.push_back(cur_pair);
    }
    std::sort(indexed_weights.begin(),
	      indexed_weights.end(),
	      pair_comp);
    // End sort of triple-matches by weight

    //std::cout << "\nT size: " << t.R_kps.size() << "\n";
    //
    double min_error = INFINITY;
    vector<cv::Mat> solution;
    vector<Matrix<double, 3, 6> > chosen;

    for (int i = 0; i < 500; i++) {
	std::vector<cv::Point2f> pts_l, pts_r, pts_p;

	vector<intpair> selection(indexed_weights);
	std::random_shuffle(selection.begin(), selection.end());

	for (int i = 0; i < selection.size(); i++) {
	    pts_l.push_back(t.L_kps[selection[i].first].pt);
	    pts_r.push_back(t.R_kps[selection[i].first].pt);
	    pts_p.push_back(t.P_kps[selection[i].first].pt);
	}
	//cv::KeyPoint::convert(t.L_kps, lkps);
	//cv::KeyPoint::convert(t.R_kps, rkps);
	//cv::KeyPoint::convert(t.P_kps, pkps);
	Matrix<double, 3, 6> pts1,pts2,pts3;
	for (int i = 0; i < 6; i++) {
	    Vector3d ptl;
	    ptl << pts_l[i].x, pts_l[i].y, 1;
	    pts1.block<3,1>(0,i) = ptl;
	    Vector3d ptr;
	    ptr << pts_r[i].x, pts_r[i].y, 1;
	    pts2.block<3,1>(0,i) = ptr;
	    Vector3d ptp;
	    ptp << pts_p[i].x, pts_p[i].y, 1;
	    pts3.block<3,1>(0,i) = ptp;
	}
	vector<Matrix<double, 3, 6> > args;
	args.push_back(pts1);
	args.push_back(pts2);
	args.push_back(pts3);
	vector<vector<Matrix<double, 3, 4> > > possible = computeTensorCandidates(args);
	int num_solutions = 0;
	// Iterate through potential solutions
	for (uint i = 0; i < possible.size(); i++) {
	    // Iterate through cameras L, R, P
	    std::vector<cv::Mat> projs, Ks, Rs, ts;
	    bool cams_valid = true;
	    for (int j = 0; j < 3; j++) {
		cv::Mat proj, K, R, t;
		cv::eigen2cv(possible[i][j], proj);
		cv::Mat innerM = proj.colRange(cv::Range(0,3));
		if(cv::determinant(innerM) != 0.0) { // change to throw out solution
		    cv::decomposeProjectionMatrix(proj, K, R, t);
		    Ks.push_back(K);
		    Rs.push_back(R);
		    ts.push_back(t);
		    projs.push_back(proj);
		} else {
		    cams_valid = false;
		}
	    }
	    if(cams_valid) {
		cv::Mat outp(4, pts_l.size(), CV_64F);
		cv::Mat lp(2, pts_l.size(), CV_64F);
		cv::Mat rp(2, pts_r.size(), CV_64F);
		cv::Mat pp(2, pts_p.size(), CV_64F);
		for (int i = 0; i < pts_l.size(); i++) {
		    lp.at<double>(0, i) = pts_l[i].x;
		    lp.at<double>(1, i) = pts_l[i].y;
		    rp.at<double>(0, i) = pts_r[i].x;
		    rp.at<double>(1, i) = pts_r[i].y;
		    pp.at<double>(0, i) = pts_p[i].x;
		    pp.at<double>(1, i) = pts_p[i].y;
		}
		cv::triangulatePoints(projs[0], projs[1], lp, rp, outp);
		double total_error = 0;
		for (int i = 0; i < outp.cols; i++) {
		    cv::Mat real_point = norm_by_index(projs[2]*outp.col(i), 2, 0);
		    cv::Mat err = real_point.rowRange(0,2) - pp.col(i);
		    total_error += (abs(err.at<double>(0,0)) + abs(err.at<double>(1,2)));
		}
		if (total_error < min_error) {
		    min_error = total_error;
		    solution.assign(projs.begin(), projs.end());
		    chosen.assign(args.begin(), args.end());
		}
		/*
		cv::Mat Kn = norm_by_index(Ks.at(0),2,2);
		if (abs(Kn.at<double>(0,1)) < min_error) {
		    min_error = abs(Kn.at<double>(0,1));
		    solution.assign(projs.begin(), projs.end());
		    chosen.assign(args.begin(), args.end());
		}
		*/
		// If there are valid solutions for all three cameras
		num_solutions++;
		//std::cout << "\nSolution #" << num_solutions << "\n";
		for (int j = 0; j < 3; j++) {
		    //std::cout << "K:" << "\n" << ppmd(Ks.at(j)) << "\n";
		    //std::cout << "R:" << "\n" << ppmd(Rs.at(j)) << "\n";
		    //std::cout << "t:" << "\n" << ppmd(ts.at(j)) << "\n\n";
		}
	    }
	}
    }
    // Return empty result while debugging
    vector<Matrix<double, 3, 4> > ret;
    for (int i = 0; i < 3; i++) {
	Matrix<double, 3, 4> toadd;
	cv::cv2eigen(solution[i], toadd);
	ret.push_back(toadd);
    }
    return ret;
}
