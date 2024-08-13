/**
 * @file cosinesimilarity.h
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.3
 * @date 2024-08-12
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef cosinesimilarity_h
#define cosinesimilarity_h

#include "../utils/helper.h"
#include "../utils/matrixfunctions.h"
#include "../utils/activationfunctions.h"
#include <string>
#include <assert.h>
#include <limits>
#include <fstream>

/**
 * @brief Performs Cosine Similarity
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
template <int channels, int input_width, int out_width, typename T>
class CosineSimilarity
{
public:
    /**
     * @brief Construct a new CosineSimilarity<channels, input_width, out_width, T>::CosineSimilarity object
     *
     * @tparam channels
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     */
    CosineSimilarity();
    /**
     * @brief Destroy the CosineSimilarity<channels, input_width, out_width, T>::CosineSimilarity object
     *
     * @tparam channels
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     */
    ~CosineSimilarity();
    /**
     * @brief Performs forwardfeed. Bias = None
     *
     * @tparam channels
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     * @param input
     * @param output
     */
    void forward(T (&input)[channels][input_width], T (&output)[channels][out_width]);
    /**
     * @brief Loadweights via infile
     *
     * @tparam channels
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     * @param infile
     */
    void loadweights(std::ifstream &infile);
    /**
     * @brief Loadweights via pathname
     *
     * @tparam channels
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     * @param pathname
     */
    void loadweights(std::string pathname);

private:
    T weights[out_width][input_width];
    T weights_T[input_width][out_width];
    T flat_matrix[input_width * out_width];
    T input_copy[channels][input_width];
};

#include "cosinesimilarity.cpp"

#endif