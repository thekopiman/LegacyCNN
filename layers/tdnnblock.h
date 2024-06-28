/**
 * @file tdnnblock.h
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.1
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef tdnnblock_h
#define tdnnblock_h

#include "conv1d.h"
#include "../utils/helper.h"
#include "../utils/activationfunctions.h"
#include "../utils/matrixfunctions.h"
#include "batchnorm1d.h"
#include <string>
#include <fstream>

// kernel, stride, channel_in, channel_out, dilation, input_width, out_width, input_pad, T
//
// Layers:
// Conv1d
// ReLU
// BatchNorm
//
// Make sure you calculate input_pad properly so that input_width == out_width

/**
 * @brief TDNNBlock
 *
 * Here the Conv1d (layer0) has been set to reflect pad
 * Here the BatchNorm1d (layer1) has been set to eval mode
 *
 * out_width = (input_width + input_pad - dilation*(kernel - 1) - 1)/stride + 1
 *
 * If you desire input_width == out_width:
 * Assume stride = 1
 *
 * then, the formula reduces to
 * input_pad = dilation * (kernal - 1)
 *
 * @tparam kernel
 * @tparam stride
 * @tparam channel_in
 * @tparam channel_out
 * @tparam dilation
 * @tparam input_width
 * @tparam out_width
 * @tparam input_pad
 * @tparam T
 */
template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, typename T>
class TDNNBlock
{
public:
    /**
     * @brief Construct a new tdnnblock<kernel, stride, channel in, channel out, dilation, input width, out width, input pad, t>::tdnnblock object
     *
     * Here the Conv1d (layer0) has been set to reflect pad
     * Here the BatchNorm1d (layer1) has been set to eval mode
     *
     * out_width = (input_width + input_pad - dilation*(kernel - 1) - 1)/stride + 1
     *
     * If you desire input_width == out_width:
     * Assume stride = 1
     *
     * then, the formula reduces to
     * input_pad = dilation * (kernal - 1)
     *
     * @tparam kernel
     * @tparam stride
     * @tparam channel_in
     * @tparam channel_out
     * @tparam dilation
     * @tparam input_width
     * @tparam out_width
     * @tparam input_pad
     * @tparam T
     */
    TDNNBlock();

    // Set weights from infile via overloading

    /**
     * @brief Loadweights via pathname
     *
     * @tparam kernel
     * @tparam stride
     * @tparam channel_in
     * @tparam channel_out
     * @tparam dilation
     * @tparam input_width
     * @tparam out_width
     * @tparam input_pad
     * @tparam T
     * @param pathname
     */
    void loadweights(std::string pathname);

    /**
     * @brief Loadweights via infile
     *
     * @tparam kernel
     * @tparam stride
     * @tparam channel_in
     * @tparam channel_out
     * @tparam dilation
     * @tparam input_width
     * @tparam out_width
     * @tparam input_pad
     * @tparam T
     * @param infile
     */
    void loadweights(std::ifstream &infile);

    /**
     * @brief Perform forward feed
     *
     * @tparam kernel
     * @tparam stride
     * @tparam channel_in
     * @tparam channel_out
     * @tparam dilation
     * @tparam input_width
     * @tparam out_width
     * @tparam input_pad
     * @tparam T
     * @param input
     * @param output
     */
    void forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_width]);

    // Debuggin purposes
    void printparameters();

    ~TDNNBlock();

private:
    Conv1d<kernel, stride, channel_in, channel_out, input_pad, dilation, input_width, out_width, T> layer0;
    BatchNorm1d<channel_out, out_width, T> layer1;
};

#include "tdnnblock.cpp"

#endif