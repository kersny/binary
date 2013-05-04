#include "odometryMath.hpp"
#include <iomanip>
#include <opencv2/core/eigen.hpp>
#include <unsupported/Eigen/NumericalDiff>
#include <unsupported/Eigen/NonLinearOptimization>

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
Eigen::Vector3d triangulatePoint(Eigen::Matrix<double, 3, 4> Pl,Eigen::Matrix<double, 3, 4> Pr,Eigen::Vector2d left_point,Eigen::Vector2d right_point)
{
    return triangulatePoint_linear_eigen(Pl,Pr,left_point,right_point);
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

Eigen::Matrix3d rotation_matrix_from_rotation_vector(Eigen::Vector3d v)
{
    Eigen::Matrix3d ret;
    cv::Mat rot_v, ret_m;
    cv::eigen2cv(v, rot_v);
    cv::Rodrigues(rot_v, ret_m);
    cv::cv2eigen(ret_m, ret);
    return ret;
}

Eigen::Vector3d rotation_vector_from_rotation_matrix(Eigen::Matrix3d m)
{
    Eigen::Vector3d ret;
    cv::Mat rot_m, rot_v;
    cv::eigen2cv(m, rot_m);
    cv::Rodrigues(rot_m, rot_v);
    cv::cv2eigen(rot_v,ret);
    return ret;
}

template<typename _Scalar, int NX=Eigen::Dynamic, int NY=Eigen::Dynamic>
struct Functor
{
  typedef _Scalar Scalar;
  enum {
    InputsAtCompileTime = NX,
    ValuesAtCompileTime = NY
  };
  typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
  typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
  typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

  int m_inputs, m_values;

  Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

};

struct Reproject : Functor<double>
{
    Reproject(BundleAdjustmentArgs a)
        : Functor<double>(6*a.num_cameras,2*a.total_points), args(a) {}
    int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
    {
        for (unsigned int j = 0; j < args.observations.size(); j++) {
            int index = args.observations[j].cameraindex;
            Eigen::Vector3d rvec = x.block<3,1>(6*index,0);
            Eigen::Vector3d tvec = x.block<3,1>(6*index, 0);
            Eigen::Matrix3d rmatrix = rotation_matrix_from_rotation_vector(rvec);
            Eigen::Matrix4d transform;
            transform.block<3,3>(0,0) = rmatrix;
            transform.block<3,1>(0,3) = tvec;
            Eigen::Vector2d lpoint = args.observations[j].left;
            Eigen::Vector2d rpoint = args.observations[j].right;
            Eigen::Vector3d world  = args.observations[j].world;
            Eigen::Vector4d world_homog;
            world_homog << world, 1.0;
            transform(3,3) = 1.0;
            Eigen::Vector3d lpoint_prime = args.project_left * transform * world_homog;
            Eigen::Vector3d rpoint_prime = args.project_right * transform * world_homog;
            fvec.block<2,1>(4*j,0) = lpoint - lpoint_prime.block<2,1>(0,0)/lpoint_prime(2,0);
            fvec.block<2,1>(4*j+2,0) = rpoint - rpoint_prime.block<2,1>(0,0)/rpoint_prime(2,0);
        }
        return 0;
    }

    private:
    BundleAdjustmentArgs args;
};
struct BundleAdjustmentFunctor
{
    BundleAdjustmentFunctor(BundleAdjustmentArgs s)
    {
        instance = new Reproject(s);
        jac = new Eigen::NumericalDiff<Reproject,Eigen::Central>(*instance);
    }
    int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
    {
        return (*instance)(x,fvec);
    }
    int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const
    {
        return jac->df(x,fjac);
    }
    ~BundleAdjustmentFunctor()
    {
        delete instance;
        delete jac;
    }
    int inputs() const { return instance->inputs(); }
    int values() const { return instance->values(); }
    Reproject* instance;
    Eigen::NumericalDiff<Reproject,Eigen::Central> *jac;
};

std::vector<Camera> bundleAdjust(BundleAdjustmentArgs args)
{
    BundleAdjustmentFunctor foo(args);
    Eigen::LevenbergMarquardt<BundleAdjustmentFunctor, double> lm(foo);
    Eigen::VectorXd bar(6*args.cameras.size());
    for (unsigned int i = 0; i < args.cameras.size(); i++) {
        bar.block<3,1>(6*i,0) = rotation_vector_from_rotation_matrix(args.cameras[i].rotation);
        bar.block<3,1>(6*i+3,0) = args.cameras[i].position;
    }
    lm.minimize(bar);
    std::vector<Camera> ret;
    for (unsigned int i = 0; i < args.cameras.size(); i++) {
        Camera c;
        c.rotation = rotation_matrix_from_rotation_vector(bar.block<3,1>(6*i,0));
        c.position = bar.block<3,1>(6*i+3,0);
        ret.push_back(c);
    }
    return ret;
}
