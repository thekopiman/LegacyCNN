/**
 * @file res2netblock.h
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.3
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef res2netblock_h
#define res2netblock_h

#include "conv1d.h"
#include "../utils/helper.h"
#include "../utils/matrixfunctions.h"
#include "tdnnblock.h"
#include <string>
#include <assert.h>

/**
 * @brief Res2NetBlock
 *
 * @tparam kernel
 * @tparam channel_in
 * @tparam channel_out
 * @tparam dilation
 * @tparam input_width
 * @tparam out_width
 * @tparam input_pad
 * @tparam scale
 * @tparam T
 */
template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, int scale, typename T>
class Res2NetBlock
{
public:
    Res2NetBlock();

    /**
     * @brief Perform Forward feed
     *
     * @tparam kernel
     * @tparam channel_in
     * @tparam channel_out
     * @tparam dilation
     * @tparam input_width
     * @tparam out_width
     * @tparam input_pad
     * @tparam scale
     * @tparam T
     * @param input
     * @param output
     */
    void forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_width]);

    /**
     * @brief Perform Forward feed
     *
     * @tparam kernel
     * @tparam channel_in
     * @tparam channel_out
     * @tparam dilation
     * @tparam input_width
     * @tparam out_width
     * @tparam input_pad
     * @tparam scale
     * @tparam T
     * @param input
     * @param lengths
     * @param output
     */
    void forward(T (&input)[channel_in][input_width], T &lengths, T (&output)[channel_out][out_width]);

    /**
     * @brief Perform Forward feed
     *
     * @tparam kernel
     * @tparam channel_in
     * @tparam channel_out
     * @tparam dilation
     * @tparam input_width
     * @tparam out_width
     * @tparam input_pad
     * @tparam scale
     * @tparam T
     * @param input
     * @param lengths
     * @param output
     */
    void forward(T (&input)[channel_in][input_width], T (&lengths)[channel_in], T (&output)[channel_out][out_width]);

    ~Res2NetBlock();

    /**
     * @brief Load the weights via pathname
     *
     * @tparam kernel
     * @tparam channel_in
     * @tparam channel_out
     * @tparam dilation
     * @tparam input_width
     * @tparam out_width
     * @tparam input_pad
     * @tparam scale
     * @tparam T
     * @param pathname
     */
    void loadweights(std::string pathname);

    /**
     * @brief Load the weights via infile
     *
     * @tparam kernel
     * @tparam channel_in
     * @tparam channel_out
     * @tparam dilation
     * @tparam input_width
     * @tparam out_width
     * @tparam input_pad
     * @tparam scale
     * @tparam T
     * @param infile
     */
    void loadweights(std::ifstream &infile);

    // For debugging purposes
    void printBlockParameters();
    void printTemp();

private:
    TDNNBlock<kernel, 1, channel_in / scale, channel_out / scale, dilation, input_width, out_width, input_pad, T> blocks[scale - 1];
    int in_channel = channel_in / scale;
    int hidden_channel = channel_out / scale;
    T y_i[channel_out / scale][out_width];
    T y[scale][channel_out / scale][out_width];
    T temp[channel_out / scale][out_width];
    T input_chunks[scale][channel_in / scale][input_width];

    /**
     * @brief Reset the values of Temp to 0.
     *
     * @tparam kernel
     * @tparam channel_in
     * @tparam channel_out
     * @tparam dilation
     * @tparam input_width
     * @tparam out_width
     * @tparam input_pad
     * @tparam scale
     * @tparam T
     */
    void resetTemp();
};

#include "res2netblock.cpp"

#endif