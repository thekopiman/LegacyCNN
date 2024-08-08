/**
 * @file asp.h
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.3
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef asp_h
#define asp_h

#include "conv1d.h"
#include "../utils/helper.h"
#include "../utils/matrixfunctions.h"
#include "../utils/activationfunctions.h"
#include "tdnnblock.h"
#include <string>
#include <assert.h>
#include <limits>
#include <fstream>

/**
 * @brief Attentive Statistical Pooling
 *
 * Here we assume: global_context = True, lengths = None
 *
 * @tparam channels
 * @tparam attention_channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 */
template <int channels, int attention_channels, int input_width, int out_width, typename T>
class ASP
{
public:
    /**
     * @brief Construct a new asp<channels, attention channels, input width, out dim, t>::asp object
     *
     * @tparam channels
     * @tparam attention_channels
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     */
    ASP();
    /**
     * @brief Destroy the asp<channels, attention channels, input width, out dim, t>::asp object
     *
     * @tparam channels
     * @tparam attention_channels
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     */
    ~ASP();
    /**
     * @brief Computes the forward feed
     *
     * @tparam channels
     * @tparam attention_channels
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     * @param input
     * @param output
     */
    void forward(T (&input)[channels][input_width], T (&output)[channels * 2][1]);
    /**
     * @brief Computes the forward feed
     *
     * Since Batch = 1, dim(lengths) == 1
     *
     * @tparam channels
     * @tparam attention_channels
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     * @param input
     * @param lengths
     * @param output
     */
    void forward(T (&input)[channels][input_width], T &lengths, T (&output)[channels * 2][1]);
    /**
     * @brief Computes the forward feed
     *
     * Since Batch = 1, dim(lengths) == 1
     *
     * @tparam channels
     * @tparam attention_channels
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     * @param input
     * @param lengths
     * @param output
     */
    void forward(T (&input)[channels][input_width], T (&lengths)[channels], T (&output)[channels * 2][1]);
    /**
     * @brief Loadweights via pathname
     *
     * @tparam channels
     * @tparam attention_channels
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     * @param pathname
     */
    void loadweights(std::string pathname);
    /**
     * @brief Loadweights via infile
     *
     * @tparam channels
     * @tparam attention_channels
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     * @param infile
     */
    void loadweights(std::ifstream &infile);

private:
    TDNNBlock<1, 1, channels * 3, attention_channels, 1, input_width, input_width, 0, T> tdnn;
    Conv1d<1, 1, attention_channels, channels, 0, 1, input_width, input_width, T> conv;
    T attn[channels * 3][input_width];
    T attn1[attention_channels][input_width];
    T attn2[channels][input_width];
    T mean[channels];
    T std[channels];
    T mask[channels][input_width];

    /**
     * @brief Compute Statistics
     *
     * @tparam channels
     * @tparam attention_channels
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     * @param input
     * @param output
     */
    void compute_statistics(T (&x)[channels][input_width], T (&m)[channels][input_width]);

    /**
     * @brief Resets the Mask to 0
     *
     * @tparam channels
     * @tparam attention_channels
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     */
    void resetMask()
};

#include "asp.cpp"

#endif