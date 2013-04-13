#include "utilities.hpp"

const uint verbosity = 3;

void debug_print(std::string str, uint level) {
    if(level <= verbosity) {
        std::cout << str;
    }
}
