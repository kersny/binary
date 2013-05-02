#include "stereoProcess.hpp"
#include "stereoBagParser.cpp"
#include "OBJParser.hpp"

int main(int argc, char** argv) {
    ros::init(argc, argv, "BINary");
    ros::NodeHandle nh;

    StereoProcess sp;

    cv::initModule_nonfree(); // stallman hates me
    std::srand((unsigned)std::time(0));


    OBJParser p = OBJParser("models/cube.obj");
    if(p.readFile()) {
        sp.modelPoints = p.getVerts();
        p.generateMeshEdges();
        sp.modelEdges = p.getEdges();
    }

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
