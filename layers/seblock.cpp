/**
 * @file seblock.cpp
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.3
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
template <int channel_in, int channel_se, int channel_out, int input_width, int out_width, typename T>
void SEBlock<channel_in, channel_se, channel_out, input_width, out_width, T>::forward(T (&input)[channel_in][input_width], T (&lengths)[channel_in], T (&output)[channel_out][out_width])
{
    resetMask();

    bool ExistNonZero = False;
    for (int i = 0; i < channel_in; i++)
    {
        if (lengths[i] > 0)
        {
            ExistNonZero = True;
        }
    }

    assert(ExistNonZero && "Make sure that lengths contain at least a Natural number to prevent division by 0.");

    // Mask here has a dim of [N, C, L] instead of the one [N, 1, L]. Where N = 1
    // This is because Hadamard Product in MatrixFunctions requires same dimensions.

    // Length to mask
    // https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/dataio/dataio.html#length_to_mask

    for (int j = 0; j < std::ceil(lengths * input_width) && j < input_width; j++)
    {
        for (int i = 0; i < channel_in; i++)
        {
            this->mask[i][j] = 1;
        }
    }

    // mask / total

    for (int i = 0; i < channel_in; i++)
    {
        for (int j = 0; j < input_width; k++)
        {
            this->mask[i][j] /= total;
        }
    }

    T total = MatrixFunctions::Sum(this->mask[0]);

    T s[channel_in][input_width];
    MatrixFunctions::HadamardProduct(input, this->mask, s);

    for (int i = 0; i < channel_in; i++)
    {
        mean[i][0] = MatrixFunctions::Sum(s[i]) / total;
    }

    layer0.forward(mean, temp);
    ActivationFunctions::ReLU(temp);
    // We reuse mean to save space
    layer1.forward(temp, mean);
    ActivationFunctions::Sigmoid(mean);
    // Reshape mean[channel_in][1] to temp2[channel_in]
    MatrixFunctions::Reshape(mean, temp2);
    MatrixFunctions::HadamardProduct(input, temp2, output);
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
 * @param lengths
 * @param output
 */
template <int channel_in, int channel_se, int channel_out, int input_width, int out_width, typename T>
void SEBlock<channel_in, channel_se, channel_out, input_width, out_width, T>::forward(T (&input)[channel_in][input_width], T &lengths, T (&output)[channel_out][out_width])
{
    T tempLengths[channel_in];
    assert(lengths > 0 && "Make sure that lengths contain at least a Natural number to prevent division by 0.");

    for (int i = 0; i < channel_in; i++)
    {
        tempLengths[i] = lengths;
    }

    forward(input, tempLengths, output);
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
template <int channel_in, int channel_se, int channel_out, int input_width, int out_width, typename T>
void SEBlock<channel_in, channel_se, channel_out, input_width, out_width, T>::resetMask()
{
    for (int i = 0; i < channel_in; i++)
    {
        for (int j = 0; j < input_width; j++)
        {
            this->mask[i][j] = 0;
        }
    }
}