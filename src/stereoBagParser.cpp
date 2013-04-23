#include "bagParser.hpp"

#define sm sensor_msgs

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

class StereoBagParser : public BagParser {
    public:
        StereoBagParser(std::string file, std::vector<std::string> arg_topics) {
            bag.open(file, rosbag::bagmode::Read);
            topics = arg_topics;
            haveL = haveR = false;
            view = new rosbag::View(bag, rosbag::TopicQuery(topics));
            myIter = (*view).begin();
            std::cout << "New bag parser for bag: " << file << "\n";
        }

        bool getNext(sm::ImageConstPtr& l_ptr, sm::ImageConstPtr& r_ptr) {
            // Keep iterating through messages until a pair 
            //   is found or there are no more messages
            while(!haveL || !haveR) {
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
                    myIter++;
                } else {
                    std::cout << "Closing bag. \n";
                    bag.close();
                    return false;
                }
            }
            l_ptr = l_img;
            r_ptr = r_img;
            haveL = haveR = false;
            std::cout << "\n\nParsed image pair. \n";
            return true;
        }

    private:
        bool haveL, haveR;
        sm::Image::ConstPtr l_img, r_img;
        rosbag::View::iterator myIter;
        rosbag::View *view;
};