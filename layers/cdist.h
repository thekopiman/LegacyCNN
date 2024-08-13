/**
 * @file cdist.h
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.3
 * @date 2024-08-12
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef cdsih_h
#define cdsih_h

#include "../utils/helper.h"
#include "../utils/matrixfunctions.h"
#include "../utils/activationfunctions.h"
#include <string>
#include <assert.h>
#include <limits>
#include <fstream>

/**
 * @brief Performs CDist
 *
 * @tparam channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 */
template <int channels, int input_width, int out_width, typename T>
class CDist
{
public:
    CDist();
    ~CDist();
    void forward(T (&input)[channels][input_width], T (&output)[channels][out_width], int p);
    void forward(T (&input)[channels][input_width], T (&output)[channels][out_width]);
    void loadweights(std::ifstream &infile);
    void loadweights(std::string pathname);

private:
    T weights[out_width][input_width];
    T weights_T[input_width][out_width];
    T flat_matrix[input_width * out_width];
    T input_copy[channels][input_width];
};

#include "cdist.cpp"

#endif