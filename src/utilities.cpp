#include "utilities.hpp"

const uint verbosity = 3;

/**
  * A debug print function to only print debug statements above a chosen importance
  *   value.
  * @param str   The string to print.
  * @param level The importance level of the given string (1 highest -> 3 lowest).
*/
void debug_print(std::string str, uint level) {
    if(level <= verbosity) {
        std::cout << str;
    }
}

// Pretty print Mat of doubles
std::string ppmd(cv::Mat m) {
    std::stringstream s;
    s << std::setiosflags(std::ios::fixed);
    s << "[";
    for(int i = 0; i < m.rows; i++) {
        if(i > 0) s << " ";
        for(int j = 0; j < m.cols; j++) {
            s << std::setw(7) << std::setprecision(4) << m.at<double>(i,j);
            if(j < m.cols-1) s << " , ";
            else s << " ; ";
        }
        s << std::endl;
    }
    s << "]";
    return s.str();
}
