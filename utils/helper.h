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

    template <size_t dim1, typename T>
    static void print(T (&input)[dim1]);

    template <size_t dim1, size_t dim2, typename T>
    static void print(T (&input)[dim1][dim2]);
};

#include "helper.cpp"

#endif