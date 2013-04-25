#include "stereoProcess.hpp"
#include "trifocalTensor.hpp"
#include "stereoBagParser.cpp"

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>

#define DRK cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS

std::string feature_type = "SIFT"; // options: "SIFT", "SURF", etc

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
    Eigen::Matrix<double, 4, 1> result = (A.transpose()*A).inverse()*(A.transpose()*b);
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
        double outlier_factor = -0.4;
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
std::vector<int> get_query_idxs(std::vector<cv::DMatch> matches) {
    std::vector<int> query_idxs;
    query_idxs.reserve(matches.size());
    for(uint i=0; i < matches.size(); i++) {
        query_idxs.push_back(matches[i].queryIdx);
    }
    return query_idxs;
}

// find a keypoint index x from a sorted vector of query indeces
int find_kp(std::vector<int> q_idxs, int x) {
    if(std::binary_search(q_idxs.begin(), q_idxs.end(), x)) {
        // R_index = Index of R keypoint in RP matches
        std::vector<int>::iterator index_iter =
            std::lower_bound(q_idxs.begin(), q_idxs.end(), x);
        int R_index = index_iter - q_idxs.begin();
        return R_index;
    } else {
        return -1;
    }
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

void StereoProcess::process_im_pair(const cv::Mat& L_mat,
                                    const cv::Mat& R_mat,
                                    ros::Time time)
{
    std::ostringstream os;
    os << "Processing image pair with timestamp: " << time << std::endl;
    debug_print(os.str(), 3);

    std::vector<cv::KeyPoint> L_kps, R_kps;
    cv::Mat L_features, R_features;

    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                L_kps = get_keypoints(L_mat);
                L_features = extract_features(L_mat, L_kps);
            }
            #pragma omp section
            {
                R_kps = get_keypoints(R_mat);
                R_features = extract_features(R_mat, R_kps);
            }
        }
    }

    std::cout << "L_kps size: " << L_kps.size() << "\n";
    std::cout << "R_kps size: " << R_kps.size() << "\n";
    std::cout << "P_kps size: " << P_kps.size() << "\n";

    std::cout << "L_features size: " << L_features.rows << "\n";
    std::cout << "R_features size: " << R_features.rows << "\n";
    std::cout << "P_features size: " << P_features.rows << "\n";

    // Do not find triple-matches on first image pair
    //  or if no features found.
    if(P_kps.size() > 0 && L_kps.size() > 0 && R_kps.size() > 0) {

        std::vector<cv::DMatch> LR_matches =
            get_matches(L_features, R_features);

        std::cout << "\nLR size: " << LR_matches.size() << "\n";

        std::vector<cv::DMatch> RP_matches =
            get_matches(R_features, P_features);

        std::cout << "\nRP size: " << RP_matches.size() << "\n";

        std::vector<cv::DMatch> PL_matches =
            get_matches(P_features, L_features);

        std::cout << "\nPL size: " << PL_matches.size() << "\n";

        // Query indeces of matches are sorted in ascending order
        std::vector<int> L_qidxs = get_query_idxs(LR_matches);
        std::vector<int> R_qidxs = get_query_idxs(RP_matches);
        std::vector<int> P_qidxs = get_query_idxs(PL_matches);

        TripleMatches t;

        for(uint i=0; i < LR_matches.size(); i++) {
            int L_kp_1 = LR_matches[i].queryIdx;
            int R_kp =   LR_matches[i].trainIdx;
            // Check if R keypoint from LR is in RP
            int RP_kp_i = find_kp(R_qidxs, R_kp);
            if(RP_kp_i != -1) {
                int P_kp = RP_matches.at(RP_kp_i).trainIdx;
                // Check if P keypoint from RP is in PL
                int PL_kp_i = find_kp(P_qidxs, P_kp);
                if(PL_kp_i != -1) {
                    int L_kp_2 = PL_matches.at(PL_kp_i).trainIdx;
                    if(L_kp_2 == L_kp_1) {
                        // Cycle is complete, same match in all 3 images
                        t.L_kps.push_back(L_kps.at(L_kp_1));
                        t.R_kps.push_back(R_kps.at(R_kp));
                        t.P_kps.push_back(P_kps.at(P_kp));

                        double weight = t.kp_weight * (
                                            L_kps.at(L_kp_1).response +
                                            R_kps.at(R_kp).response +
                                            P_kps.at(P_kp).response) +
                                        t.match_dist_weight * (
                                            LR_matches[i].distance +
                                            RP_matches.at(RP_kp_i).distance +
                                            PL_matches.at(PL_kp_i).distance);

                        t.weights.push_back(weight);
                    }
                }
            }
        }
        std::cout << "TripleMatches size: " << t.R_kps.size() << "\n";
	cv::Mat Kl = (cv::Mat_<double>(3,3) << 1107.58877335145,0,703.563442850518,0,1105.93566117489,963.193789785819,0,0,1);
	cv::Mat Kr = (cv::Mat_<double>(3,3) << 1104.28764692449,0,761.642398493953,0,1105.31682336766,962.344514230255,0,0,1);
	cv::Mat C = (cv::Mat_<double>(3,4) << 1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0);
	cv::Mat PoseL = (cv::Mat_<double>(4,4) << 1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0);
	cv::Mat Ldist_coeff = (cv::Mat_<double>(1,5) << -0.0306, 0.053, 0.0020, 0.0014, 0.000);
	cv::Mat Rdist_coeff = (cv::Mat_<double>(1,5) << -0.0243, 0.0448, 0.0027, 0.0023, 0.000);
	//cv::Mat PoseR = (cv::Mat_<double>(4,4) << 0.999971932224562,-0.00732216763241206,-0.0015876473912136,-554.348268227282, 0.00729111397179107,0.999797530643968,-0.0187546627608496,-0.435011047094735,0.001724650725893,0.0187425606411105,0.999822855310123,-0.789271765090486,0,0,0,1);
	cv::Mat Pl = Kl * C * PoseL;
	//cv::Mat Pr = Kl * C * PoseR;
	cv::Mat Pr = (cv::Mat_<double>(3,4) << 1105.57021914223,6.18934957543074,759.754258185686,-612760.0875376,9.71869909913803, 1123.12983099782,941.444195743573,-1240.37638207625, 0.001724650725893,0.0187425606411105,0.999822855310123,-0.789271765090486);
	std::vector<cv::Point3f> pts3;
	std::vector<cv::Point2f> prev_points_d, left_points_d, right_points_d, prev_points, left_points, right_points;
	for (int i = 0; i < t.L_kps.size(); i++) {
	    left_points_d.push_back(t.L_kps[i].pt);
	    right_points_d.push_back(t.R_kps[i].pt);
	    prev_points_d.push_back(t.P_kps[i].pt);
	}
	cv::undistortPoints(left_points_d, left_points, Kl, Ldist_coeff);
	cv::undistortPoints(right_points_d, right_points, Kr, Rdist_coeff);
	cv::undistortPoints(prev_points_d, prev_points, Kl, Ldist_coeff);
	for (int i = 0; i < left_points.size(); i++) {
	    //cout << left << " " << right << endl;
	    cv::Mat pt = triangulatePoint(Pl,Pr,left_points[i],right_points[i]);
	    cv::Point3f actual;
	    actual.x = pt.at<double>(0,0);
	    actual.y = pt.at<double>(1,0);
	    actual.z = pt.at<double>(2,0);
	    pts3.push_back(actual);
	}
	cv::Mat tvec(3,1,CV_64F);
	cv::Mat rvec(3,1,CV_64F);

	vector<int> inliers;
	cv::solvePnPRansac(pts3, prev_points_d, Kl, Ldist_coeff, rvec, tvec);
	cv::Mat R;
	cv::Rodrigues(rvec, R);
	cout << R << endl << tvec << endl;

        cv::Mat stiched = make_mono_image(L_mat, R_mat, t.L_kps, t.R_kps);
        sized_show(stiched, 0.25, "MONO IMAGE");

        // features / matches of triple matches
        cv::Mat L2_features = extract_features(L_mat, t.L_kps);
        cv::Mat R2_features = extract_features(R_mat, t.R_kps);

        std::vector<cv::DMatch> LR2_matches =
            get_matches(L2_features, R2_features);

        // Display matches
        cv::Mat img_matches;
        cv::drawMatches(L_mat, t.L_kps, R_mat, t.R_kps,
                        LR2_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        sized_show(img_matches, 0.25, "MATCHES");

        cv::Mat L_out, R_out, P_out;
        cv::drawKeypoints(L_mat, t.L_kps, L_out, cv::Scalar(255, 0, 0), DRK);
        cv::drawKeypoints(R_mat, t.R_kps, R_out, cv::Scalar(255, 0, 0), DRK);
        cv::drawKeypoints(P_mat, t.P_kps, P_out, cv::Scalar(255, 0, 0), DRK);
        sized_show(L_out, 0.25, "LEFT");
        sized_show(R_out, 0.25, "RIGHT");
        sized_show(P_out, 0.25, "PREV");
        cv::waitKey(10);
    }

    L_features.copyTo(P_features);
    P_kps = L_kps;
    L_mat.copyTo(P_mat);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "BINary");
    ros::NodeHandle nh;

    StereoProcess sp;

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
