/**
 * @file seblock.h
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.3
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef seblock_h
#define seblock_h

#include "conv1d.h"
#include "../utils/helper.h"
#include "../utils/activationfunctions.h"
#include "../utils/matrixfunctions.h"
#include "tdnnblock.h"
#include <string>
#include <assert.h>
#include <fstream>

/**
 * @brief SEBlock
 *
 * We assume lengths = None
 *
 * @tparam channel_in
 * @tparam channel_se
 * @tparam channel_out
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 */
template <int channel_in, int channel_se, int channel_out, int input_width, int out_width, typename T>
class SEBlock
{
public:
    /**
     * @brief Construct a new seblock<channel in, channel se, channel out, input width, out width, t>::seblock object
     *
     * The Conv1d layers (layer0 and layer1) have reflect padding by default.
     *
     * @tparam channel_in
     * @tparam channel_se
     * @tparam channel_out
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     */
    SEBlock();
    /**
     * @brief Perform forward feed
     *
     * @tparam channel_in
     * @tparam channel_se
     * @tparam channel_out
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     * @param input
     * @param output
     */
    void forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_width]);

    /**
     * @brief Perform forward feed
     *
     * @tparam channel_in
     * @tparam channel_se
     * @tparam channel_out
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     * @param input
     * @param lengths
     * @param output
     */
    void forward(T (&input)[channel_in][input_width], T &lengths, T (&output)[channel_out][out_width]);
    /**
     * @brief Perform forward feed
     *
     * @tparam channel_in
     * @tparam channel_se
     * @tparam channel_out
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     * @param input
     * @param lengths
     * @param output
     */
    void forward(T (&input)[channel_in][input_width], T (&lengths)[channel_in], T (&output)[channel_out][out_width]);
    ~SEBlock();

    /**
     * @brief Load weights via pathname
     *
     * @tparam channel_in
     * @tparam channel_se
     * @tparam channel_out
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     * @param pathname
     */
    void loadweights(std::string pathname);

    /**
     * @brief Load weights via infile
     *
     * @tparam channel_in
     * @tparam channel_se
     * @tparam channel_out
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     * @param infile
     */
    void loadweights(std::ifstream &infile);

private:
    Conv1d<1, 1, channel_in, channel_se, 0, 1, 1, 1, T> layer0;
    T temp[channel_se][1];
    Conv1d<1, 1, channel_se, channel_out, 0, 1, 1, 1, T> layer1;
    T mean[channel_in][1];
    T temp2[channel_out];
    T mask[channel_in][input_width];
    T total[channel_in];

    /**
     * @brief Resets the Mask to 0
     *
     * @tparam channel_in
     * @tparam channel_se
     * @tparam channel_out
     * @tparam input_width
     * @tparam out_width
     * @tparam T
     */
    void
    resetMask();
};

#include "seblock.cpp"

#endif
