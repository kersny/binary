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
    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols; j++) {
           // s << std::setw(5) << m(i,j);
        }
        s << std::endl;
    }
    return s.str();
}
