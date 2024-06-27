/**
 * @file basicblock.h
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief Basic Block with Conv1d, Relu and BatchNorm1
 * @version 0.1
 * @date 2024-06-27
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef basiccnn_h
#define basiccnn_h

#include "conv1d.h"
#include "../utils/helper.h"
#include "../utils/activationfunctions.h"
#include "../utils/matrixfunctions.h"
#include "batchnorm1d.h"
#include <string>
#include <fstream>

/**
 * @brief This code snippet defines a template class `BasicBlock` that represents a basic block in a convolutional neural network (CNN).
 * Which consist of Sequential([Conv1d, Relu, BatchNorm1])
 *
 * @tparam kernel
 * @tparam stride
 * @tparam channel_in
 * @tparam channel_out
 * @tparam pad
 * @tparam dilation
 * @tparam input_width
 * @tparam out_dim
 * @tparam T
 */

template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_dim, typename T>
class BasicBlock
{
public:
    BasicBlock();

    /**
     * @brief Load the weighs via pathname
     *
     * @tparam kernel
     * @tparam stride
     * @tparam channel_in
     * @tparam channel_out
     * @tparam pad
     * @tparam dilation
     * @tparam input_width
     * @tparam out_dim
     * @tparam T
     * @param infile
     */
    void loadweights(std::string pathname);

    /**
     * @brief Load the weighs via infile
     *
     * @tparam kernel
     * @tparam stride
     * @tparam channel_in
     * @tparam channel_out
     * @tparam pad
     * @tparam dilation
     * @tparam input_width
     * @tparam out_dim
     * @tparam T
     * @param infile
     */
    void loadweights(std::ifstream &infile);

    /**
     * @brief The `forward` function takes a 2D input array `input` of type `T` with dimensions
     * `[channel_in][input_width]` and a 2D output array `output` of type `T` with dimensions
     * `[channel_out][out_dim]`.
     *
     * @tparam kernel
     * @tparam stride
     * @tparam channel_in
     * @tparam channel_out
     * @tparam pad
     * @tparam dilation
     * @tparam input_width
     * @tparam out_dim
     * @tparam T
     * @param input
     * @param output
     */
    void forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim]);

    /**
     * @brief Destroy the BasicBlock<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_dim, T>::BasicBlock object
     *
     * @tparam kernel
     * @tparam stride
     * @tparam channel_in
     * @tparam channel_out
     * @tparam pad
     * @tparam dilation
     * @tparam input_width
     * @tparam out_dim
     * @tparam T
     */
    ~BasicBlock();

private:
    Conv1d<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_dim, T> layer0;
    BatchNorm1d<channel_out, out_dim, T> layer1;
};

#include "basicblock.cpp"

#endif