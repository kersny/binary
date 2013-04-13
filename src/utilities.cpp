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
