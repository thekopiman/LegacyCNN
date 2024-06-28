/**
 * @file basicblock.cpp
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief Basic Block with Conv1d, Relu and BatchNorm1
 * @version 0.1
 * @date 2024-06-27
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "basicblock.h"

/**
 * @brief Construct a new BasicBlock<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_width, T>::BasicBlock object.
 * Conv1d (layer0) will have normal padding mode.
 * BatchNorm1d (layer1) will be in eval mode.
 *
 * @tparam kernel
 * @tparam stride
 * @tparam channel_in
 * @tparam channel_out
 * @tparam pad
 * @tparam dilation
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 */
template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_width, typename T>
BasicBlock<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_width, T>::BasicBlock() : layer1(1){};

/**
 * @brief Destroy the BasicBlock<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_width, T>::BasicBlock object
 *
 * @tparam kernel
 * @tparam stride
 * @tparam channel_in
 * @tparam channel_out
 * @tparam pad
 * @tparam dilation
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 */
template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_width, typename T>
BasicBlock<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_width, T>::~BasicBlock(){};

/**
 * @brief Load the weights from pathname
 *
 * @tparam kernel
 * @tparam stride
 * @tparam channel_in
 * @tparam channel_out
 * @tparam pad
 * @tparam dilation
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 * @param pathname
 */
template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_width, typename T>
void BasicBlock<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_width, T>::loadweights(std::string pathname)
{
    std::ifstream infile(pathname, std::ios::binary);
    if (!infile)
    {
        std::cout << "Error opening file!" << std::endl;
        return;
    }
    // Read dimensions
    layer0.loadweights(infile);
    layer1.loadweights(infile);
    infile.close();
}

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
 * @tparam out_width
 * @tparam T
 * @param infile
 */
template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_width, typename T>
void BasicBlock<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_width, T>::loadweights(std::ifstream &infile)
{
    if (!infile)
    {
        std::cout << "Error opening file!" << std::endl;
        return;
    }
    // Read dimensions
    layer0.loadweights(infile);
    layer1.loadweights(infile);
}

/**
 * @brief The `forward` function takes a 2D input array `input` of type `T` with dimensions
 * `[channel_in][input_width]` and a 2D output array `output` of type `T` with dimensions
 * `[channel_out][out_width]`.
 *
 * @tparam kernel
 * @tparam stride
 * @tparam channel_in
 * @tparam channel_out
 * @tparam pad
 * @tparam dilation
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 * @param input
 * @param output
 */
template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_width, typename T>
void BasicBlock<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_width, T>::forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_width])
{
    this->layer0.forward(input, output);
    ActivationFunctions::ReLU<channel_out, out_width, T>(output);
    this->layer1.forward(output, output);
}
