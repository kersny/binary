#include "utilities.hpp"

const uint verbosity = 2;

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

// Normalize a matrix of doubles to fix one entry to be 1
cv::Mat norm_by_index(cv::Mat m, int r, int c) {
    double norm_val = m.at<double>(r,c);
    cv::Mat normed = m / norm_val;
    return normed;
}

void sized_show(cv::Mat img, double size_mult, std::string window_name) {
    cv::Mat resized_img = 
        cv::Mat::zeros(img.rows * size_mult, img.cols * size_mult, img.type());
    cv::resize(img, resized_img, resized_img.size());
    cv::imshow(window_name, resized_img);
}

// Pretty print Mat of doubles
std::string ppmd(cv::Mat m) {
    // Find width needed to fit largest number in matrix
    double abs_max_val = 0.0;
    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols; j++) {
            double abs_cur = abs(m.at<double>(i,j));
            if(abs_cur > abs_max_val)
                abs_max_val = abs_cur;
        }
    }
    int width;
    int prec = 4;
    if(abs_max_val < 10) width = prec + 2 + 2;
    else if(abs_max_val >= 10) width = prec + 2 + 3; // decimals, dot, negative sign, digits
    else if(abs_max_val >= 100) width = prec + 2 + 4; 
    else width = prec + 2 + 5;

    // Pretty print matrix accordingly
    std::stringstream s;
    s << std::setiosflags(std::ios::fixed);
    s << "[";
    for(int i = 0; i < m.rows; i++) {
        if(i > 0) s << " ";
        for(int j = 0; j < m.cols; j++) {
            s << std::setw(width) << std::setprecision(prec) << m.at<double>(i,j);
            if(j < m.cols-1) s << " , ";
        }
        if(i < m.rows-1) s << " ; " << std::endl;
    }
    s << "]";
    return s.str();
}
