#ifndef helper_h
#define helper_h

#include <string>
#include <fstream>
#include <cmath>
#include <iostream>
#include <assert.h>

class Helper
{
public:
    template <size_t rows, size_t cols, typename T>
    static void readInputs(std::string filename, T (&inputs)[rows][cols]);
};

#include "helper.cpp"

#endif