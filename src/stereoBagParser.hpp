#include "bagParser.hpp"

#define sm sensor_msgs

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

class StereoBagParser: public BagParser {
    public:
        StereoBagParser(std::string file, std::vector<std::string> arg_topics) {
            bag.open(file, rosbag::bagmode::Read);
            topics = arg_topics;
            haveL = haveR = false;
            view = new rosbag::View(bag, rosbag::TopicQuery(topics));
            rosbag::View::iterator myIter = (*view).begin();
        }

        bool getNext(sm::ImageConstPtr& l_ptr, sm::ImageConstPtr& r_ptr) {
            if(myIter != (*view).end()) {
                rosbag::MessageInstance const m = *myIter;
                if (m.getTopic() == topics.at(0)) {
                    l_img = m.instantiate<sm::Image>();
                    haveL = true;
                }
                if (m.getTopic() == topics.at(1)) {
                    r_img = m.instantiate<sm::Image>();
                    haveR = true;
                }
                if(haveL && haveR) {
                    l_ptr = l_img;
                    r_ptr = r_img;
                    haveL = haveR = false;
                }
                myIter++;
                return true;
            } else {
                bag.close();
                return false;
            }
        }

    private:
        bool haveL, haveR;
        sm::Image::ConstPtr l_img, r_img;
        rosbag::View::iterator myIter;
        rosbag::View *view;
};