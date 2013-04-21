#ifndef BAG_PARSER
#define BAG_PARSER

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>

#include <string>

class BagParser {
    public:
        virtual ~BagParser();
        virtual bool getNext();

    protected:
        rosbag::Bag bag;
        std::vector<std::string> topics;
};

#endif