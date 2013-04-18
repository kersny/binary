#include "stereoProcess.hpp"
//BAD TODO: FIX to be not horribly bad
#include "trifocalTensor.cpp"

#define DRK cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS

StereoProcess::StereoProcess()
{
    L_channel = "/stereo/left/image_raw";
    R_channel = "/stereo/right/image_raw";
    max_im_pairs = 20;
}

std::vector<cv::KeyPoint> StereoProcess::get_keypoints(cv::Mat img)
{
    // Detect SIFT keypoints in both images
    debug_print("Detecting SIFT keypoints.\n", 3);
    static cv::SiftDescriptorExtractor detector;
    std::vector<cv::KeyPoint> kps;
    detector.detect(img, kps);
    return kps;
}

cv::Mat StereoProcess::extract_features(cv::Mat img, std::vector<cv::KeyPoint> kps)
{
    // Extract SIFT features
    debug_print("Extracting SIFT features.\n", 3);
    static cv::SiftDescriptorExtractor extractor;
    cv::Mat features;
    extractor.compute(img, kps, features );
    return features;
}

std::vector<cv::DMatch> get_matches(cv::Mat L_features, cv::Mat R_features)
{
    // Find feature matches between two images
    debug_print("Matching SIFT features.\n", 3);
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
        double outlier_factor = -0.3;
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
std::vector<int> get_query_idxs(std::vector<cv::DMatch> matches)
{
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

class TripleMatches {
    public:
        std::vector<cv::KeyPoint> L_kps;
        std::vector<cv::KeyPoint> R_kps;
        std::vector<cv::KeyPoint> P_kps;
};

void StereoProcess::process_im_pair(const cv::Mat& L_mat,
                                    const cv::Mat& R_mat,
                                    ros::Time t)
{
    std::ostringstream os;
    os << "Processing image pair with timestamp: " << t << std::endl;
    debug_print(os.str(), 3);

    std::vector<cv::KeyPoint> L_kps = get_keypoints(L_mat);
    std::vector<cv::KeyPoint> R_kps = get_keypoints(R_mat);

    cv::Mat L_features = extract_features(L_mat, L_kps);
    cv::Mat R_features = extract_features(R_mat, R_kps);

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
                    }
                }
            }
        }

        std::cout << "\nT size: " << t.R_kps.size() << "\n";
	std::vector<Eigen::Matrix<double, 3, 6> > matchPoints;
	Eigen::Matrix<double, 3, 6> points1;
	Eigen::Matrix<double, 3, 6> points2;
	Eigen::Matrix<double, 3, 6> points3;
	for (int i = 0; i < 6; i++) {
	    Eigen::Vector3d pt1;
	    pt1 << t.R_kps[i].pt.x, t.R_kps[i].pt.y, 1;
	    points1.block<3,1>(0,i) = pt1;
	    Eigen::Vector3d pt2;
	    pt2 << t.L_kps[i].pt.x, t.L_kps[i].pt.y, 1;
	    points2.block<3,1>(0,i) = pt2;
	    Eigen::Vector3d pt3;
	    pt3 << t.P_kps[i].pt.x, t.P_kps[i].pt.y, 1;
	    points3.block<3,1>(0,i) = pt3;
	}
	matchPoints.push_back(points1);
	matchPoints.push_back(points2);
	matchPoints.push_back(points3);
	std::vector<std::vector<Matrix<double, 3, 4> > > ret = computeTensorCandidates(matchPoints);
	for (int i = 0; i < ret.size(); i++) {
	    for (int j = 0; j < 3; j++) {
		cv::Mat proj, K, R, t;
		cv::eigen2cv(ret[i][j], proj);
		cv::decomposeProjectionMatrix(proj, K, R, t);
		std::cout << K << endl;
		std::cout << R << endl;
		std::cout << t << endl;
	    }
	}

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
        cv::Mat matches_small;
        matches_small = cv::Mat::zeros(img_matches.rows / 3, img_matches.cols / 3, 16);
        cv::resize(img_matches, matches_small, matches_small.size());
        cv::namedWindow("Matches", CV_WINDOW_AUTOSIZE);
        cv::imshow("Matches" , matches_small);

        cv::Mat L_out, R_out, P_out;
        cv::drawKeypoints(L_mat, t.L_kps, L_out, cv::Scalar(255, 0, 0), DRK);
        cv::drawKeypoints(R_mat, t.R_kps, R_out, cv::Scalar(255, 0, 0), DRK);
        cv::drawKeypoints(P_mat, t.P_kps, P_out, cv::Scalar(255, 0, 0), DRK);
        cv::Mat L_small, R_small, P_small;
        L_small = cv::Mat::zeros(L_out.rows / 4, L_out.cols / 4, CV_8UC1);
        R_small = cv::Mat::zeros(R_out.rows / 4, R_out.cols / 4, CV_8UC1);
        P_small = cv::Mat::zeros(P_out.rows / 4, P_out.cols / 4, CV_8UC1);
        cv::resize(L_out, L_small, L_small.size());
        cv::resize(R_out, R_small, R_small.size());
        cv::resize(P_out, P_small, P_small.size());
        cv::namedWindow("LEFT",  CV_WINDOW_AUTOSIZE);
        cv::namedWindow("RIGHT", CV_WINDOW_AUTOSIZE);
        cv::namedWindow("PREV", CV_WINDOW_AUTOSIZE);
        cv::imshow("LEFT" , L_small);
        cv::imshow("RIGHT", R_small);
        cv::imshow("PREV" , P_small);
        cv::waitKey(10);
    }

    P_features = L_features;
    P_kps = L_kps;
    L_mat.copyTo(P_mat);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "BINary");
    ros::NodeHandle nh;

    StereoProcess sp;

    mf::Subscriber<sm::Image> L_sub(nh, sp.L_channel, 1);
    mf::Subscriber<sm::Image> R_sub(nh, sp.R_channel, 1);

    typedef mf::sync_policies::ApproximateTime<sm::Image, sm::Image> MySyncPolicy;
    mf::Synchronizer<MySyncPolicy> sync( \
        MySyncPolicy(sp.max_im_pairs), L_sub, R_sub);

    sync.registerCallback(
        boost::bind(&StereoProcess::im_pair_callback, &sp, _1, _2));

    ros::spin();

    return 0;
}
