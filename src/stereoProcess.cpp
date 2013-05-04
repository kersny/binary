#include "stereoProcess.hpp"
#include "stereoBagParser.cpp"
#include "OBJParser.hpp"
#include "odometryMath.hpp"
#include "frame.hpp"

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>
#include <cmath>
#include <algorithm>

#define DRK cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
#define FEATURE_DETECTOR "SIFT"
#define DESCRIPTOR_EXTRACTOR "SIFT"


StereoProcess::StereoProcess() {
    L_channel = "/stereo/left/image_raw";
    R_channel = "/stereo/right/image_raw";
    max_im_pairs = 20;
    position = Eigen::Vector3d::Zero();
    orientation = Eigen::Matrix3d::Identity();
    worldRot = Eigen::Matrix3d::Identity();
    modelOrigin = Eigen::Vector3d(3000, 0, 0);
    worldPos = Eigen::Vector3d(0, 0, 0);
    std::cout << "Using " << FEATURE_DETECTOR << " Feature Extractor" << std::endl;
    extractor = cv::DescriptorExtractor::create(FEATURE_DETECTOR);
    std::cout << "Using " << DESCRIPTOR_EXTRACTOR << " Feature Descriptor" << std::endl;
    detector = cv::FeatureDetector::create(DESCRIPTOR_EXTRACTOR);
}

std::vector<cv::KeyPoint> StereoProcess::get_keypoints(cv::Mat img) {
    // Detect keypoints in both images
    debug_print("Detecting keypoints.\n", 3);
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
        for(unsigned int i=0; i < matches.size(); i++) {
            dist_sum += matches[i].distance;
        }
        double dist_mean = dist_sum / matches.size();
        // Find standard deviation
        double sum_sq_diff = 0.0;
        for(unsigned int i=0; i < matches.size(); i++) {
            double cur_diff = matches[i].distance - dist_mean;
            sum_sq_diff += cur_diff * cur_diff;
        }
        double std_dev = sqrt(sum_sq_diff / matches.size());
        // Refine matches by throwing out outliers
        // outlier_factor = number of standard deviations
        //                  above mean to consider an outlier
        double outlier_factor = -0.4;
        cv::vector<cv::DMatch> good_matches;
        for(unsigned int i=0; i < matches.size(); i++) {
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
    for(unsigned int i=0; i < matches.size(); i++) {
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
    std::vector< std::vector<cv::KeyPoint> >
StereoProcess::get_circular_matches(std::vector< std::vector<cv::KeyPoint> > all_pts,
        std::vector< cv::Mat> all_features)
{
    unsigned int n = all_pts.size();
    // Get n cycle matches 0->1 , 1->2, ... (n-1)->0
    std::vector< std::vector<cv::DMatch> > cycle_matches;
    // Query indeces of matches are sorted in ascending order
    std::vector< std::vector<int> > cycle_qidxs;
    for(unsigned int i = 0; i < n; i++) {
        int a = i;         // current image's whose features are being matched
        int b = (i+1) % n; // index of a's neighbor in cycle
        cv::Mat a_fts = all_features[a];
        cv::Mat b_fts = all_features[b];
        cycle_matches.push_back( get_matches(a_fts, b_fts));
        cycle_qidxs.push_back( get_query_idxs(cycle_matches[i]));
    }
    // Extract only keypoints that can be matched through the whole cycle
    std::vector< std::vector<cv::KeyPoint> > cycle_kps;
    std::vector<int> weights;
    for(unsigned int x = 0; x < n; x++) {
        // initialize solution vectors
        std::vector<cv::KeyPoint> tmpKPv;
        cycle_kps.push_back(tmpKPv);
    }
    // Loop through all keypoints in image 0 from match 0->1
    for(unsigned int i = 0; i < cycle_matches[0].size(); i++) {
        // The query index of a found point in a cyclic match
        //  for each of the original keypoint sets
        std::vector< int > kp_qidxs;
        kp_qidxs.reserve(n);
        int start_query_val = cycle_matches[0][i].queryIdx;
        kp_qidxs[0] = start_query_val;
        // Attempt to follow this keypoint through all images
        //  back to itself
        int start_train_val = cycle_matches[0][i].trainIdx;
        bool lost = false; // able to follow matches
        int cur_train_val = start_train_val;
        unsigned int x = 1; // x is index of next image
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
            int kp_quality = 0.0;
            for(unsigned int x = 0; x < n; x++) {
                cv::KeyPoint cur_pt = all_pts[x][ kp_qidxs[x] ];
                cycle_kps[x].push_back(cur_pt);
                kp_quality += cur_pt.response;
            }
            weights.push_back(-1.0 * kp_quality);
            // TODO: add match weights in and use keypoint weights
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

    for( unsigned int i = 0; i < R_kps.size(); i++ ) {
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


///////
void drawLine( cv::Mat img, cv::Point start, cv::Point end )
{
  int thickness = 2;
  int lineType = 8;
  cv::line( img,
            start, end,
            cv::Scalar( 255, 0, 0 ),
            thickness, lineType );
}

void drawDot( cv::Mat img, cv::Point center )
{
    int thickness = -1;
    int lineType = 8;

    cv::circle( img,
            center,
            5.0,
            cv::Scalar( 0, 0, 255 ),
            thickness, lineType );
}

void StereoProcess::process_im_pair(const cv::Mat& CL_mat,
        const cv::Mat& CR_mat,
        ros::Time c_time)
{
    std::cout << bundle_frames.size() << std::endl;
    if (bundle_frames.size() >= 10) {
        std::cout << "Bundle Adjusting" << std::endl;
        std::vector< std::vector<cv::KeyPoint> > BA_pts;
        std::vector<cv::Mat> BA_fts;
        for (unsigned int i = 0; i < bundle_frames.size(); i++) {
            BA_pts.push_back(bundle_frames[i].L_kps);
            BA_fts.push_back(bundle_frames[i].L_features);
        }
        for (int i = (int)bundle_frames.size() - 1; i >= 0; i--) {
            BA_pts.push_back(bundle_frames[i].R_kps);
            BA_fts.push_back(bundle_frames[i].R_features);
        }
        std::vector< std::vector<cv::KeyPoint> > BA_good_pts;
        BA_good_pts = get_circular_matches(BA_pts, BA_fts);
        bundle_frames.clear();
    }
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
        std::vector< std::vector<cv::KeyPoint> > all_pts;
        all_pts.push_back(CL_kps);
        all_pts.push_back(CR_kps);
        all_pts.push_back(PL_kps);
        all_pts.push_back(PR_kps);
        std::vector<cv::Mat> all_fts;
        all_fts.push_back(CL_features);
        all_fts.push_back(CR_features);
        all_fts.push_back(PL_features);
        all_fts.push_back(PR_features);
        std::vector< std::vector<cv::KeyPoint> > good_pts;
        good_pts = get_circular_matches(all_pts, all_fts);
        std::cout << "GoodPoints size: " << good_pts[0].size() << "\n";

        //cv::Mat Kl = (cv::Mat_<double>(3,3) << 1107.58877335145,0,703.563442850518,0,1105.93566117489,963.193789785819,0,0,1);
        //cv::Mat Kr = (cv::Mat_<double>(3,3) << 1104.28764692449,0,761.642398493953,0,1105.31682336766,962.344514230255,0,0,1);
        //cv::Mat C = (cv::Mat_<double>(3,4) << 1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0);
        //cv::Mat PoseL = (cv::Mat_<double>(4,4) << 1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0);
        //cv::Mat PoseR = (cv::Mat_<double>(4,4) << 1.0000,-0.0073,-0.0016, -554.3483, 0.0073,0.9998,-0.0188,-0.4350, 0.0017,0.0187,0.9998,-0.7893, 0,0,0,1.0000);
        //cv::Mat Ldist_coeff = (cv::Mat_<double>(1,5) << -0.0305748283698362, 0.0530084757712889, 0.00198169725147652, 0.0013820669430398, 0);
        //cv::Mat Rdist_coeff = (cv::Mat_<double>(1,5) << -0.0243498347962812, 0.0447656953196109, 0.0026529511902253, 0.00225483859237588, 0);
        Eigen::Matrix<double, 3, 4> Pl,Pr;
        Pl << 1107.58877335145, 0, 703.563442850518, 0,
              0, 1105.93566117489, 963.193789785819, 0,
              0, 0, 1, 0;
        Pr << 1105.57021914223,6.18934957543074,759.754258185686,-612760.0875376,
              9.71869909913803, 1123.12983099782,941.444195743573,-1240.37638207625,
              0.001724650725893,0.0187425606411105,0.999822855310123,-0.789271765090486;
        std::vector<Eigen::Vector3d> pts3_now, pts3_prev;
        std::vector<Eigen::Vector2d> prev_left_points, prev_right_points, left_points, right_points;
        for (unsigned int i = 0; i < good_pts[0].size(); i++) {
            Eigen::Vector2d tmp;
            tmp(0,0) = good_pts[0][i].pt.x;
            tmp(1,0) = good_pts[0][i].pt.y;
            left_points.push_back(tmp);
            tmp(0,0) = good_pts[1][i].pt.x;
            tmp(1,0) = good_pts[1][i].pt.y;
            right_points.push_back(tmp);
            tmp(0,0) = good_pts[2][i].pt.x;
            tmp(1,0) = good_pts[2][i].pt.y;
            prev_left_points.push_back(tmp);
            tmp(0,0) = good_pts[3][i].pt.x;
            tmp(1,0) = good_pts[3][i].pt.y;
            prev_right_points.push_back(tmp);
        }

	for (unsigned int i = 0; i < left_points.size(); i++) {
	    Eigen::Vector3d pt_now = triangulatePoint(Pl,Pr,left_points[i],right_points[i]);
            Eigen::Vector3d pt_prev = triangulatePoint(Pl,Pr,prev_left_points[i],prev_right_points[i]);
	    pts3_now.push_back(pt_now);
	    pts3_prev.push_back(pt_prev);
	}
        std::pair<Eigen::Matrix3d,Eigen::Vector3d> ans = computeOrientationRansac(pts3_now, pts3_prev);
        Eigen::Matrix3d R_final = Eigen::Matrix3d::Zero();
        Eigen::Vector3d T_final = Eigen::Vector3d::Zero();
        R_final = ans.first;
        T_final = ans.second;
        position += T_final;
        orientation = R_final * orientation;
        Frame f;
        f.L_kps = CL_kps;
        f.R_kps = CR_kps;
        f.L_features = CL_features;
        f.R_features = CR_features;
        f.Orientation_world = orientation;
        f.Translation_world = position;
        bundle_frames.push_back(f);
        std::cout << "Cur Rotation: \n" << R_final << std::endl << "dT: \n\n" << T_final << std::endl;
        std::cout << "Orientation: \n" << orientation << std::endl << "T: \n" << position << std::endl;

        //cv::Mat stiched = make_mono_image(CL_mat, CR_mat, good_pts[0], good_pts[1]);
        //sized_show(stiched, 0.25, "MONO IMAGE");

        cv::Mat PL_out, PR_out;
        //cv::drawKeypoints(CL_mat, good_pts[0], CL_out, cv::Scalar(255, 0, 0), DRK);
        //cv::drawKeypoints(CR_mat, good_pts[1], CR_out, cv::Scalar(255, 0, 0), DRK);
        cv::Mat CL_out(CL_mat.size(), CV_8UC3);
        cv::cvtColor(CL_mat, CL_out, CV_GRAY2RGB);
        cv::Mat CR_out(CR_mat.size(), CV_8UC3);
        cv::cvtColor(CR_mat, CR_out, CV_GRAY2RGB);

        Eigen::Matrix4d B_from_W; // W is world frame, B is z-forward camera basis
        B_from_W << 0, -1,  0, 0,    \
                    0,  0, -1, 2000, \
                    1,  0,  0, 0,    \
                    0,  0,  0, 1;
        // negative y -> positive x
        // negative z -> positive y
        // positive x -> positive z
        // camera is roughly 2m (2000 mm) above ground

        Eigen::Matrix4d CI_from_B; // B is frame of cameras with Z exactly forward
        // CI is initial frame of the cameras
        double roll = -30.0 * 3.14159 / 180.0;
        // From Z-forward base frame to tilted down cameras is a negative roll
        CI_from_B << 1,  0,          0,          0,    \
                     0,  cos(roll),  sin(roll),  0,    \
                     0, -sin(roll),  cos(roll),  0,    \
                     0,  0,          0,          1;

        std::vector<cv::Point> modelPts2d_L, modelPts2d_R; // points of model in images
        // compute vertices projections in both current images
        for(unsigned int i = 0 ; i < modelPoints.size(); i++) {
            Eigen::Vector4d model_vert;
            model_vert.block<3,1>(0,0) = 100.0 * modelPoints[i]; // 1m cube
            model_vert.block<3,1>(0,0) += modelOrigin; // center to origin
            model_vert.block<3,1>(0,0) -= worldPos; // move relative to our real position
            model_vert(3, 0) = 1; // homogenous
            // place vertex in reference frame of initial camera
            Eigen::Matrix4d CC_from_CI = Eigen::Matrix4d::Identity();
            CC_from_CI.block<3,3>(0,0) = orientation.inverse();
            model_vert = CC_from_CI * (CI_from_B * (B_from_W * model_vert));
            if(model_vert(2, 0) < 0) {
                // Batman style points can suddenly appear behind camera too due to P
                cv::Point sentinelPt = cv::Point(-10000, -10000);
                modelPts2d_L.push_back(sentinelPt);
                modelPts2d_R.push_back(sentinelPt);
            } else {
                // Get vector of image coordinates for vertices of model
                Eigen::Vector3d im_pt_homog_L = Pl * model_vert;
                im_pt_homog_L /= im_pt_homog_L(2,0);
                Eigen::Vector3d im_pt_homog_R = Pr * model_vert;
                im_pt_homog_R /= im_pt_homog_R(2,0);
                cv::Point im_ptL = cv::Point(im_pt_homog_L(0,0),
                                             im_pt_homog_L(1,0));
                cv::Point im_ptR = cv::Point(im_pt_homog_R(0,0),
                                             im_pt_homog_R(1,0));
                modelPts2d_L.push_back(im_ptL);
                modelPts2d_R.push_back(im_ptR);
            }
        }
        // draw model edges in both images
        for(unsigned int i = 0 ; i < modelEdges.size(); i++) {
            int x = modelEdges[i].first;
            int y = modelEdges[i].second;
            if(modelPts2d_L[x].x != -10000) // no edges for fake points
                drawLine(CL_out, modelPts2d_L[x], modelPts2d_L[y]);
            if(modelPts2d_R[x].x != -10000)
                drawLine(CR_out, modelPts2d_R[x], modelPts2d_R[y]);
        }
        for(unsigned int i = 0 ; i < modelPts2d_L.size(); i++) {
            drawDot(CL_out, modelPts2d_L[i]);
            drawDot(CR_out, modelPts2d_R[i]);
        }

        Eigen::Matrix3d B_from_CI_R = CI_from_B.block<3,3>(0,0).inverse();
        Eigen::Matrix3d W_from_B_R = B_from_W.block<3,3>(0,0).inverse();
        // Get current odometry position into world frame
        worldPos = W_from_B_R * (B_from_CI_R * position);
        std::cout << "World pos: \n" << worldPos << "\n";

        cv::drawKeypoints(PL_mat, good_pts[2], PL_out, cv::Scalar(255, 0, 0), DRK);
        cv::drawKeypoints(PR_mat, good_pts[3], PR_out, cv::Scalar(255, 0, 0), DRK);
        sized_show(CL_out, 0.4, "CURR LEFT");
        sized_show(CR_out, 0.4, "CURR RIGHT");
        sized_show(PL_out, 0.4, "PREV LEFT");
        sized_show(PR_out, 0.4, "PREV RIGHT");
        int delay = 10;
        char input = cv::waitKey(delay);
        if(delay >= 100) { // Laggy game mode!
            if(input == 'w') {
                modelOrigin(0,0) += 100;
            } else if(input == 's') {
                modelOrigin(0,0) -= 100;
            } else if(input == 'a') {
                modelOrigin(1,0) += 100;
            } else if(input == 'd') {
                modelOrigin(1,0) -= 100;
            } else if(input == 'x') {
                modelOrigin(2,0) -= 100;
            } else if(input == 32) { // space bar
                modelOrigin(2,0) += 100;
            }
        }
    }
    CL_features.copyTo(PL_features);
    CR_features.copyTo(PR_features);
    PL_kps = CL_kps;
    PR_kps = CR_kps;
    CL_mat.copyTo(PL_mat);
    CR_mat.copyTo(PR_mat);
}
