#include "stereoProcess.hpp"
#include "trifocalTensor.hpp"
#include "stereoBagParser.cpp"
#include "unsupported/Eigen/NonLinearOptimization"

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>

#define DRK cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS

std::string feature_type = "SIFT"; // options: "SIFT", "SURF", etc
class ReprojFunc
{
    public:
	Eigen::Matrix<double, 3, 4> _P1;
	Eigen::Matrix<double, 3, 4> _P2;
	Eigen::Vector3d _x1;
	Eigen::Vector3d _x2;
	ReprojFunc(const Eigen::Matrix<double, 3, 4> P1, const Eigen::Matrix<double, 3, 4> P2, const Eigen::Vector3d x1, const Eigen::Vector3d x2)
	    : _P1(P1),_P2(P2),_x1(x1),_x2(x2) {};
	int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
	{
	    Eigen::Vector3d x1_prime = _P1*x;
	    Eigen::Vector3d x2_prime = _P2*x;
	    fvec(0) = x1_prime(0) - _x1(0);
	    fvec(1) = x1_prime(1) - _x1(1);
	    fvec(2) = x1_prime(2) - _x1(2);
	    fvec(3) = x2_prime(0) - _x2(0);
	    fvec(4) = x2_prime(1) - _x2(1);
	    fvec(5) = x2_prime(2) - _x2(2);
	    return 0;
	}
	int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const
	{
	    Eigen::VectorXd row1(4);
	    Eigen::VectorXd row2(4);
	    Eigen::VectorXd row3(4);
	    Eigen::VectorXd row4(4);
	    Eigen::VectorXd row5(4);
	    Eigen::VectorXd row6(4);
	    row1 << _P1(0,0), _P1(0,1), _P1(0,2), _P1(0,3);
	    row2 << _P1(1,0), _P1(1,1), _P1(1,2), _P1(1,3);
	    row3 << _P1(2,0), _P1(2,1), _P1(2,2), _P1(2,3);
	    row4 << _P2(0,0), _P2(0,1), _P2(0,2), _P2(0,3);
	    row5 << _P2(1,0), _P2(1,1), _P2(1,2), _P2(1,3);
	    row6 << _P2(2,0), _P2(2,1), _P2(2,2), _P2(2,3);
	    fjac.row(0) = row1;
	    fjac.row(1) = row2;
	    fjac.row(2) = row3;
	    fjac.row(3) = row4;
	    fjac.row(4) = row5;
	    fjac.row(5) = row6;
	    return 0;
	}

	int inputs() const { return 4; }// inputs is the dimension of x.
	int values() const { return 6; } // "values" is the number of f_i and
};

cv::Mat triangulatePoint(cv::Mat Pl,cv::Mat Pr,cv::Point2f left_point,cv::Point2f right_point)
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
    Eigen::VectorXd result = (A.transpose()*A).inverse()*(A.transpose()*b);
    result = result/result(3,0);
    ReprojFunc func(Ple,Pre,lp,rp);
    Eigen::LevenbergMarquardt<ReprojFunc, double> lm(func);
    lm.minimize(result);
    result = result/result(3,0);
    cv::Mat ret(4,1,CV_64F);
    cv::eigen2cv(result,ret);
    return ret;
}

StereoProcess::StereoProcess() {
    L_channel = "/stereo/left/image_raw";
    R_channel = "/stereo/right/image_raw";
    max_im_pairs = 20;
}

std::vector<cv::KeyPoint> StereoProcess::get_keypoints(cv::Mat img) {
    // Detect keypoints in both images
    debug_print("Detecting keypoints.\n", 3);
    cv::Ptr<cv::FeatureDetector> detector;
    detector = cv::FeatureDetector::create(feature_type);
    std::vector<cv::KeyPoint> kps;
    detector->detect(img, kps);
    return kps;
}

cv::Mat StereoProcess::extract_features(cv::Mat img,
                                        std::vector<cv::KeyPoint> kps)
{
    // Extract features
    debug_print("Extracting features.\n", 3);
    cv::Mat features;
    cv::Ptr<cv::DescriptorExtractor> extractor;
    extractor = cv::DescriptorExtractor::create(feature_type);
    extractor->compute(img, kps, features);
    return features;
}

std::vector<cv::DMatch> get_matches(cv::Mat L_features, cv::Mat R_features) {
    // Find feature matches between two images
    debug_print("Matching features.\n", 3);
    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(L_features, R_features, matches);

    if(matches.size() == 0) {
        debug_print("No matches found!\n", 2);
        return matches;
    } else {
        debug_print("Filtering for best matches.\n", 3);
        // Compute mean match distance
        double dist_sum = 0.0;
        for(uint i=0; i < matches.size(); i++) {
            dist_sum += matches[i].distance;
        }
        double dist_mean = dist_sum / matches.size();
        // Find standard deviation
        double sum_sq_diff = 0.0;
        for(uint i=0; i < matches.size(); i++) {
            double cur_diff = matches[i].distance - dist_mean;
            sum_sq_diff += cur_diff * cur_diff;
        }
        double std_dev = sqrt(sum_sq_diff / matches.size());
        // Refine matches by throwing out outliers
        // outlier_factor = number of standard deviations
        //                  above mean to consider an outlier
        double outlier_factor = -0.45;
        cv::vector<cv::DMatch> good_matches;
        for(uint i=0; i < matches.size(); i++) {
            if(matches[i].distance < dist_mean + outlier_factor * std_dev) {
                good_matches.push_back(matches[i]);
            }
        }
        return good_matches;
    }
}

// Return the sorted vector of query indeces from a vector of matches
std::vector<int> StereoProcess::get_query_idxs(std::vector<cv::DMatch> matches) {
    std::vector<int> query_idxs;
    query_idxs.reserve(matches.size());
    for(uint i=0; i < matches.size(); i++) {
        query_idxs.push_back(matches[i].queryIdx);
    }
    return query_idxs;
}

// find a keypoint index x from a sorted vector of query indeces
int StereoProcess::find_kp(std::vector<int> q_idxs, int x) {
    if(std::binary_search(q_idxs.begin(), q_idxs.end(), x)) {
        // B_index = Index of B keypoint in RP matches
        std::vector<int>::iterator index_iter =
            std::lower_bound(q_idxs.begin(), q_idxs.end(), x);
        int B_index = index_iter - q_idxs.begin();
        return B_index;
    } else {
        return -1;
    }
}

// Get only keypoints that match correctly across keypoint sets from n images
//  given keypoints and corresponding features from all frames
vector< vector<cv::KeyPoint> > 
    StereoProcess::get_circular_matches(vector< vector<cv::KeyPoint> > all_pts,
                                        vector< cv::Mat> all_features) 
{
    uint n = all_pts.size();
    // Get n cycle matches 0->1 , 1->2, ... (n-1)->0
    vector< vector<cv::DMatch> > cycle_matches;
    // Query indeces of matches are sorted in ascending order
    vector< vector<int> > cycle_qidxs;
    for(uint i = 0; i < n; i++) {
        int a = i;         // current image's whose features are being matched
        int b = (i+1) % n; // index of a's neighbor in cycle
        cv::Mat a_fts = all_features[a];
        cv::Mat b_fts = all_features[b];
        cycle_matches.push_back( get_matches(a_fts, b_fts));
        cycle_qidxs.push_back( get_query_idxs(cycle_matches[i]));
    }
    // Extract only keypoints that can be matched through the whole cycle
    vector< vector<cv::KeyPoint> > cycle_kps;
    for(uint x = 0; x < n; x++) {
        // initialize solution vectors
        vector<cv::KeyPoint> tmp;
        cycle_kps.push_back(tmp);
    }
    // Loop through all keypoints in image 0 from match 0->1
    for(uint i = 0; i < cycle_matches[0].size(); i++) {
        // The query index of a found point in a cyclic match 
        //  for each of the original keypoint sets
        vector< int > kp_qidxs;
        kp_qidxs.reserve(n);
        int start_query_val = cycle_matches[0][i].queryIdx;
        kp_qidxs[0] = start_query_val;
        // Attempt to follow this keypoint through all images
        //  back to itself
        int start_train_val = cycle_matches[0][i].trainIdx;
        bool lost = false; // able to follow matches
        int cur_train_val = start_train_val;
        uint x = 1; // x is index of next image
        while( !lost && x < n) {
            int next_index = find_kp(cycle_qidxs[x], cur_train_val);
            if(next_index == -1) {
                lost = true;
            } else {
                kp_qidxs[x] = cur_train_val;
                cur_train_val = cycle_matches[x][next_index].trainIdx;
                x++;
            }
        }
        // if final match ended at original position in image 0
        //  then populate the solution with the matched points
        if(!lost && cur_train_val == start_query_val) {
            for(uint x = 0; x < n; x++) {
                cv::KeyPoint cur_pt = all_pts[x][ kp_qidxs[x] ];
                cycle_kps[x].push_back(cur_pt);
                // TODO: add match weights in
            }
        }
    }
    return cycle_kps;
}

// Takes the left and right image, and ordered matching keypoints from each
//  and produces the stiched together monocular version of the stereo images
cv::Mat make_mono_image(cv::Mat L_mat, cv::Mat R_mat,
                      std::vector<cv::KeyPoint> L_kps,
                      std::vector<cv::KeyPoint> R_kps)
{
    std::vector<cv::Point2f> L_pts;
    std::vector<cv::Point2f> R_pts;

    for( uint i = 0; i < R_kps.size(); i++ ) {
        //-- Get the keypoints from the good matches
        L_pts.push_back( L_kps[i].pt );
        R_pts.push_back( R_kps[i].pt );
    }
    cv::Mat H = cv::findHomography( L_pts, R_pts, CV_RANSAC );
    std::cout << "\nHomography : \n" << ppmd(H) << "\n";

    cv::Mat L_warped = cv::Mat::zeros(L_mat.rows, L_mat.cols, CV_8UC1);
    cv::warpPerspective(L_mat, L_warped, H, L_warped.size());

    cv::Mat stiched(L_mat.rows, L_mat.cols, CV_8UC1);
    int blend_dist = 100; // 1/2 the width of blending area in center
    cv::Rect L_good_ROI(L_mat.cols / 2 + blend_dist, 0,
                        L_mat.cols / 2 - blend_dist, L_mat.rows);
    cv::Rect R_good_ROI(0, 0, R_mat.cols / 2 - blend_dist, R_mat.rows);
    cv::Mat partLW = cv::Mat(L_warped, L_good_ROI);
    cv::Mat partR = cv::Mat(R_mat, R_good_ROI);
    partLW.copyTo(stiched(L_good_ROI));
    partR.copyTo(stiched(R_good_ROI));

    cv::Rect blend_ROI(L_mat.cols / 2 - blend_dist, 0,
                       2 * blend_dist, L_mat.rows);
    cv::Mat midLW(L_warped, blend_ROI);
    cv::Mat midR(R_mat, blend_ROI);

    cv::Mat center_blend(blend_ROI.size(), CV_8UC1);
    cv::addWeighted( midR, 0.5, midLW, 0.5, 0.0, center_blend);
    center_blend.copyTo(stiched(blend_ROI));

    return stiched;
}

void StereoProcess::process_im_pair(const cv::Mat& CL_mat,
                                    const cv::Mat& CR_mat,
                                    ros::Time c_time)
{
    std::ostringstream os;
    os << "Processing image pair with timestamp: " << c_time << std::endl;
    debug_print(os.str(), 3);

    std::vector<cv::KeyPoint> CL_kps, CR_kps;
    cv::Mat CL_features, CR_features;

    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                CL_kps = get_keypoints(CL_mat);
                CL_features = extract_features(CL_mat, CL_kps);
            }
            #pragma omp section
            {
                CR_kps = get_keypoints(CR_mat);
                CR_features = extract_features(CR_mat, CR_kps);
            }
        }
    }

    std::cout << "CL_kps size: " << CL_kps.size() << "\n";
    std::cout << "CR_kps size: " << CR_kps.size() << "\n";
    std::cout << "PL_kps size: " << PL_kps.size() << "\n";
    std::cout << "PR_kps size: " << PR_kps.size() << "\n";

    // Do not find triple-matches on first image pair
    //  or if no features found.
    if(CL_kps.size() == 0 || CR_kps.size() == 0 ||
       PL_kps.size() == 0 || PR_kps.size() == 0 )
    {
        std::cout << "Error! Not enough keypoints!";
    } else {
        vector< vector<cv::KeyPoint> > all_pts;
        all_pts.push_back(CL_kps);
        all_pts.push_back(CR_kps);
        all_pts.push_back(PL_kps);
        all_pts.push_back(PR_kps);
        vector<cv::Mat> all_fts;
        all_fts.push_back(CL_features);
        all_fts.push_back(CR_features);
        all_fts.push_back(PL_features);
        all_fts.push_back(PR_features);
        vector< vector<cv::KeyPoint> > good_pts;
        good_pts = get_circular_matches(all_pts, all_fts);
        std::cout << "GoodPoints size: " << good_pts[0].size() << "\n";

        cv::Mat Kl = (cv::Mat_<double>(3,3) << 1107.58877335145,0,703.563442850518,0,1105.93566117489,963.193789785819,0,0,1);
        cv::Mat Kr = (cv::Mat_<double>(3,3) << 1104.28764692449,0,761.642398493953,0,1105.31682336766,962.344514230255,0,0,1);
        cv::Mat C = (cv::Mat_<double>(3,4) << 1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0);
        cv::Mat PoseL = (cv::Mat_<double>(4,4) << 1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0);
        cv::Mat Ldist_coeff = (cv::Mat_<double>(1,5) << -0.0306, 0.053, 0.0020, 0.0014, 0.000);
        cv::Mat Rdist_coeff = (cv::Mat_<double>(1,5) << -0.0243, 0.0448, 0.0027, 0.0023, 0.000);
        //cv::Mat PoseR = (cv::Mat_<double>(4,4) << 0.999971932224562,-0.00732216763241206,-0.0015876473912136,-554.348268227282, 0.00729111397179107,0.999797530643968,-0.0187546627608496,-0.435011047094735,0.001724650725893,0.0187425606411105,0.999822855310123,-0.789271765090486,0,0,0,1);
        cv::Mat Pl = Kl * C * PoseL;
        //cv::Mat Pr = Kl * C * PoseR;
        cv::Mat Pr = (cv::Mat_<double>(3,4) << \
                1105.57021914223,6.18934957543074,759.754258185686,-612760.0875376, \
                9.71869909913803, 1123.12983099782,941.444195743573,-1240.37638207625, \
                0.001724650725893,0.0187425606411105,0.999822855310123,-0.789271765090486);
        std::vector<cv::Point3f> pts3;
        std::vector<cv::Point2f> prev_L_pts_d, prev_R_pts_d, curr_L_pts_d, curr_R_pts_d, 
                                 prev_L_pts, prev_R_pts, curr_L_pts, curr_R_pts;
        for (uint i = 0; i < good_pts[0].size(); i++) {
            curr_L_pts_d.push_back( good_pts[0][i].pt);
            curr_R_pts_d.push_back( good_pts[1][i].pt);
            prev_L_pts_d.push_back( good_pts[2][i].pt);
            prev_R_pts_d.push_back( good_pts[3][i].pt);
        }
        cv::undistortPoints(curr_L_pts_d, curr_L_pts, Kl, Ldist_coeff);
        cv::undistortPoints(curr_R_pts_d, curr_R_pts, Kr, Rdist_coeff);
        cv::undistortPoints(prev_L_pts_d, prev_L_pts, Kl, Ldist_coeff);
        cv::undistortPoints(prev_R_pts_d, prev_R_pts, Kr, Rdist_coeff);
        uint n = curr_L_pts.size();
        // solve for linear 3DOF translation by minimizing point correspondence error
        MatrixXd A = MatrixXd::Zero(3*n, 3);
        MatrixXd b = MatrixXd::Zero(3*n, 1);
        // note X = [ x ; y ; z ]
        for (uint i = 0; i <n; i++) {
            cv::Mat curr_pt = triangulatePoint(Pl,Pr,curr_L_pts[i],curr_R_pts[i]);
            cv::Mat prev_pt = triangulatePoint(Pl,Pr,prev_L_pts[i],prev_R_pts[i]);
            A(3*i + 0, 0) = 1;
            A(3*i + 1, 1) = 1;
            A(3*i + 2, 2) = 1;
            b(3*i + 0, 0) = prev_pt.at<double>(0,0) - curr_pt.at<double>(0,0);
            b(3*i + 1, 0) = prev_pt.at<double>(1,0) - curr_pt.at<double>(1,0);
            b(3*i + 2, 0) = prev_pt.at<double>(2,0) - curr_pt.at<double>(2,0);
            //cout << "point delta: \n" << ppmd(curr_pt - prev_pt);
            // cv::Point3f actual;
            // actual.x = curr_pt.at<double>(0,0);
            // actual.y = curr_pt.at<double>(1,0);
            // actual.z = curr_pt.at<double>(2,0);
            // pts3.push_back(actual);
            //std::cout << "c \n" << curr_pt;
            //std::cout << prev_pt;
        }
 //       cout << "\na: \n" << A;
 //       cout << "\nb: \n" << b;
        MatrixXd t = A.jacobiSvd( ComputeFullU | ComputeFullV ).solve(b);
        cout << "\nt: \n" << t / 1000.0;
        POS += t;
        cout << "\nPOS: \n" << POS / 1000.0;
        // cv::Mat tvec(3,1,CV_64F);
        // cv::Mat rvec;

        //vector<int> inliers;
        // int iterations = 100;
        // cv::solvePnPRansac(pts3, prev_points_d, Kl, Ldist_coeff, 
        //                        rvec, tvec, false, iterations);
        // cv::Mat R,Rt;
        // cv::Rodrigues(rvec, R);
        // cv::transpose(R,Rt);
        // cout << ppmd(R) << endl << ppmd(-Rt*tvec) << endl;

        //cv::Mat stiched = make_mono_image(L_mat, R_mat, t.L_kps, t.R_kps);
        //sized_show(stiched, 0.25, "MONO IMAGE");

        // features / matches of triple matches
        //cv::Mat L2_features = extract_features(L_mat, good_pts[0]);
        //cv::Mat R2_features = extract_features(R_mat, good_pts[1]);

        //std::vector<cv::DMatch> LR2_matches =
        //    get_matches(L2_features, R2_features);

        // Display matches
        // cv::Mat img_matches;
        // cv::drawMatches(L_mat, good_pts[0], R_mat, good_pts[1],
        //                 LR2_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
        //                 std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        // sized_show(img_matches, 0.4, "MATCHES");

        cv::Mat CL_out, CR_out, PL_out, PR_out;
        cv::drawKeypoints(CL_mat, good_pts[0], CL_out, cv::Scalar(255, 0, 0), DRK);
        cv::drawKeypoints(CR_mat, good_pts[1], CR_out, cv::Scalar(255, 0, 0), DRK);
        cv::drawKeypoints(PL_mat, good_pts[2], PL_out, cv::Scalar(255, 0, 0), DRK);
        cv::drawKeypoints(PR_mat, good_pts[3], PR_out, cv::Scalar(255, 0, 0), DRK);
        sized_show(CL_out, 0.4, "CURR LEFT");
        sized_show(CR_out, 0.4, "CURR RIGHT");
        sized_show(PL_out, 0.4, "PREV LEFT");
        sized_show(PR_out, 0.4, "PREV RIGHT");
        cv::waitKey(10);
    }
    CL_features.copyTo(PL_features);
    CR_features.copyTo(PR_features);
    PL_kps = CL_kps;
    PR_kps = CR_kps;
    CL_mat.copyTo(PL_mat);
    CR_mat.copyTo(PR_mat);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "BINary");
    ros::NodeHandle nh;

    StereoProcess sp;

    //tmp
    sp.POS = MatrixXd::Zero(3, 1);

    cv::initModule_nonfree(); // stallman hates me

    std::cout << "\nInitialized program, using " << feature_type << " features.\n";

    if(argc >= 2) { // If given a bag file to parse
        std::vector<std::string> topics;
        topics.push_back(std::string(sp.L_channel));
        topics.push_back(std::string(sp.R_channel));

        StereoBagParser parser = StereoBagParser(argv[1], topics);
        sm::ImageConstPtr l_img, r_img;
        while(ros::ok() && parser.getNext(l_img, r_img)) {
            sp.im_pair_callback(l_img, r_img);
        }
    } else { // In real-time listening mode using subscribers
        mf::Subscriber<sm::Image> L_sub(nh, sp.L_channel, 1);
        mf::Subscriber<sm::Image> R_sub(nh, sp.R_channel, 1);
        typedef mf::sync_policies::ApproximateTime<sm::Image, sm::Image> MySyncPolicy;
        mf::Synchronizer<MySyncPolicy> sync( \
            MySyncPolicy(sp.max_im_pairs), L_sub, R_sub);
        sync.registerCallback(
            boost::bind(&StereoProcess::im_pair_callback, &sp, _1, _2));

        ros::spin();
    }

    return 0;
}
