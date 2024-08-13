/**
 * @file euclidean.h
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.3
 * @date 2024-08-13
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef Euclidean_h
#define Euclidean_h

#include "../utils/helper.h"
#include "../utils/matrixfunctions.h"
#include "../utils/activationfunctions.h"
#include <string>
#include <assert.h>
#include <limits>
#include <fstream>

/**
 * @brief Performs Euclidean
 *
 * F.linear(input.normalised(), weights.normalised())
 *
 * We assume Bias = None
 *
 * @tparam channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 */
template <int channels, int input_width, typename T>
class Euclidean
{
public:
    Euclidean();
    ~Euclidean();
    void forward(T (&input)[channels][input_width], T (&output)[channels]);
    void loadweights(std::ifstream &infile);
    void loadweights(std::string pathname);

private:
    T weights[channels][input_width];
    T flat_matrix[input_width * channels];
};

#include "euclidean.cpp"

#endif