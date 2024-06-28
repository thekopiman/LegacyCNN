/**
 * @file seblock.cpp
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.1
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "seblock.h"

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
template <int channel_in, int channel_se, int channel_out, int input_width, int out_width, typename T>
SEBlock<channel_in, channel_se, channel_out, input_width, out_width, T>::SEBlock() : layer0(3), layer1(3)
{
    // Sanity check
    assert(input_width == out_width);
    assert(channel_in == channel_out);

    // std::cout << "SEBlock is initialised" << std::endl;
}

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
template <int channel_in, int channel_se, int channel_out, int input_width, int out_width, typename T>
void SEBlock<channel_in, channel_se, channel_out, input_width, out_width, T>::forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_width])
{
    MatrixFunctions::Mean(input, mean);
    layer0.forward(mean, temp);
    ActivationFunctions::ReLU(temp);
    // We reuse mean to save space
    layer1.forward(temp, mean);
    ActivationFunctions::Sigmoid(mean);
    // Reshape mean[channel_in][1] to temp2[channel_in]
    MatrixFunctions::Reshape(mean, temp2);
    MatrixFunctions::HadamardProduct(input, temp2, output);
}

template <int channel_in, int channel_se, int channel_out, int input_width, int out_width, typename T>
SEBlock<channel_in, channel_se, channel_out, input_width, out_width, T>::~SEBlock() {}

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
template <int channel_in, int channel_se, int channel_out, int input_width, int out_width, typename T>
void SEBlock<channel_in, channel_se, channel_out, input_width, out_width, T>::loadweights(std::string pathname)
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
}

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
template <int channel_in, int channel_se, int channel_out, int input_width, int out_width, typename T>
void SEBlock<channel_in, channel_se, channel_out, input_width, out_width, T>::loadweights(std::ifstream &infile)
{
    layer0.loadweights(infile);
    layer1.loadweights(infile);
}