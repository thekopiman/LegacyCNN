/**
 * @file seres2netblock.h
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.3
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef seres2netblock_h
#define seres2netblock_h

#include "conv1d.h"
#include "../utils/helper.h"
#include "../utils/matrixfunctions.h"
#include "tdnnblock.h"
#include "res2netblock.h"
#include "seblock.h"
#include <string>
#include <assert.h>
#include <fstream>

// kernel, channel_in, channel_out, dilation, input_width, out_width, input_pad, res2net_scale,channel_se, T
/**
 * @brief SERes2NetBlock
 *
 * @tparam kernel
 * @tparam channel_in
 * @tparam channel_out
 * @tparam dilation
 * @tparam input_width
 * @tparam out_width
 * @tparam input_pad
 * @tparam res2net_scale
 * @tparam channel_se
 * @tparam T
 */
template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, int res2net_scale, int channel_se, typename T>
class SERes2NetBlock
{
public:
    /**
     * @brief Construct a new seres2netblock<kernel, channel in, channel out, dilation, input width, out width, input pad, res2net scale, channel se, t>::seres2netblock object
     *
     * Make sure that input_width == out_width
     *
     * @tparam kernel
     * @tparam channel_in
     * @tparam channel_out
     * @tparam dilation
     * @tparam input_width
     * @tparam out_width
     * @tparam input_pad
     * @tparam res2net_scale
     * @tparam channel_se
     * @tparam T
     */
    SERes2NetBlock();

    /**
     * @brief Perform forward feed
     *
     * @tparam kernel
     * @tparam channel_in
     * @tparam channel_out
     * @tparam dilation
     * @tparam input_width
     * @tparam out_width
     * @tparam input_pad
     * @tparam res2net_scale
     * @tparam channel_se
     * @tparam T
     * @param input
     * @param lengths
     * @param output
     */
    void forward(T (&input)[channel_in][input_width], T (&lengths)[channel_out], T (&output)[channel_out][out_width]);
    /**
     * @brief Perform forward feed
     *
     * @tparam kernel
     * @tparam channel_in
     * @tparam channel_out
     * @tparam dilation
     * @tparam input_width
     * @tparam out_width
     * @tparam input_pad
     * @tparam res2net_scale
     * @tparam channel_se
     * @tparam T
     * @param input
     * @param output
     */
    void forward(T (&input)[channel_in][input_width], T &lengths, T (&output)[channel_out][out_width]);
    /**
     * @brief Perform forward feed
     *
     * @tparam kernel
     * @tparam channel_in
     * @tparam channel_out
     * @tparam dilation
     * @tparam input_width
     * @tparam out_width
     * @tparam input_pad
     * @tparam res2net_scale
     * @tparam channel_se
     * @tparam T
     * @param input
     * @param output
     */
    void forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_width]);
    ~SERes2NetBlock();

    /**
     * @brief Loadweights via pathname
     *
     * @tparam kernel
     * @tparam channel_in
     * @tparam channel_out
     * @tparam dilation
     * @tparam input_width
     * @tparam out_width
     * @tparam input_pad
     * @tparam res2net_scale
     * @tparam channel_se
     * @tparam T
     * @param pathname
     */
    void loadweights(std::string pathname);
    /**
     * @brief Loadweights via infile
     *
     * @tparam kernel
     * @tparam channel_in
     * @tparam channel_out
     * @tparam dilation
     * @tparam input_width
     * @tparam out_width
     * @tparam input_pad
     * @tparam res2net_scale
     * @tparam channel_se
     * @tparam T
     * @param infile
     */
    void loadweights(std::ifstream &infile);

private:
    TDNNBlock<1, 1, channel_in, channel_out, 1, input_width, out_width, 0, T> tdnn1;
    T x1[channel_out][out_width];
    Res2NetBlock<kernel, channel_out, channel_out, dilation, input_width, out_width, input_pad, res2net_scale, T> res2net;
    TDNNBlock<1, 1, channel_out, channel_out, 1, input_width, out_width, 0, T> tdnn2;
    SEBlock<channel_out, channel_se, channel_out, input_width, out_width, T> seblock;
    Conv1d<1, 1, channel_in, channel_out, 0, 1, input_width, out_width, T> shortcut;
    T residual[channel_out][out_width];
};

#include "seres2netblock.cpp"

#endif
