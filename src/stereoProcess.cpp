#include "stereoProcess.hpp"

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
    cv::SiftDescriptorExtractor detector;
    std::vector<cv::KeyPoint> kps;
    detector.detect(img, kps);
    return kps;
}

cv::Mat StereoProcess::extract_features(cv::Mat img, std::vector<cv::KeyPoint> kps)
{
    // Extract SIFT features
    debug_print("Extracting SIFT features.\n", 3);
    cv::SiftDescriptorExtractor extractor;
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
        double outlier_factor = -0.5;
        cv::vector<cv::DMatch> good_matches;
        for(uint i=0; i < matches.size(); i++) {
            if(matches[i].distance < dist_mean + outlier_factor * std_dev) {
                good_matches.push_back(matches[i]);
            }
        }
        return good_matches;
    }
}

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

    std::vector<cv::DMatch> matches = get_matches(L_features, R_features);

    // Display matches
    cv::Mat img_matches;
    cv::drawMatches(L_mat, L_kps, R_mat, R_kps,
                    matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    cv::Mat matches_small;
    matches_small = cv::Mat::zeros(img_matches.rows / 3, img_matches.cols / 3, 16);
    cv::resize(img_matches, matches_small, matches_small.size());
    cv::namedWindow("Matches", CV_WINDOW_AUTOSIZE); 
    cv::imshow("Matches" , matches_small);

    cv::Mat L_out, R_out;
    cv::drawKeypoints(L_mat, L_kps, L_out, cv::Scalar(255, 0, 0), DRK);
    cv::drawKeypoints(R_mat, R_kps, R_out, cv::Scalar(255, 0, 0), DRK);
    cv::Mat L_small, R_small;
    L_small = cv::Mat::zeros(L_out.rows / 4, L_out.cols / 4, CV_8UC1);
    R_small = cv::Mat::zeros(R_out.rows / 4, R_out.cols / 4, CV_8UC1);
    cv::resize(L_out, L_small, L_small.size());
    cv::resize(R_out, R_small, R_small.size());
    cv::namedWindow("LEFT",  CV_WINDOW_AUTOSIZE); 
    cv::namedWindow("RIGHT", CV_WINDOW_AUTOSIZE); 
    cv::imshow("LEFT" , L_small);
    cv::imshow("RIGHT", R_small);
    cv::waitKey(5);
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