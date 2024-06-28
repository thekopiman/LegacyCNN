/**
 * @file tdnnblock.cpp
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.1
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "tdnnblock.h"
#include <fstream>

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
template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, typename T>
TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_width, input_pad, T>::TDNNBlock() : layer0(3), layer1(1){};

template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, typename T>
TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_width, input_pad, T>::~TDNNBlock(){};

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
template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, typename T>
void TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_width, input_pad, T>::forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_width])
{
    layer0.forward(input, output);
    ActivationFunctions::ReLU<channel_out, out_width, T>(output);
    layer1.forward(output, output);
};

// We assume this layout
//  layer0 dim - weights
//  layer0 weights
//  layer0 dim - bias
//  layer0 bias
//  layer1 dim - weights
//  layer1 weights
//  layer1 dim - bias
//  layer1 bias
//  layer1 dim - running mean
//  layer1 mean
//  layer1 dim - running variance
//  layer1 var

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
template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, typename T>
void TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_width, input_pad, T>::loadweights(std::string pathname)
{
    std::ifstream infile(pathname, std::ios::binary);
    if (!infile)
    {
        std::cout << "Error opening file!" << std::endl;
        return;
    }
    layer0.loadweights(infile);
    layer1.loadweights(infile);

    infile.close();
};

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
template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, typename T>
void TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_width, input_pad, T>::loadweights(std::ifstream &infile)
{
    if (!infile)
    {
        std::cout << "Error opening file!" << std::endl;
        return;
    }
    layer0.loadweights(infile);
    layer1.loadweights(infile);
};

template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, typename T>
void TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_width, input_pad, T>::printparameters()
{
    std::cout << "kernel " << kernel << std::endl;
    std::cout << "stride " << stride << std::endl;
    std::cout << "channel_in " << channel_in << std::endl;
    std::cout << "channel_out " << channel_out << std::endl;
    std::cout << "dilation " << dilation << std::endl;
    std::cout << "input_pad " << input_pad << std::endl;
    // std::cout << "Kernel " << kernel << std::endl;
}