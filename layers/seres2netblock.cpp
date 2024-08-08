/**
 * @file seres2netblock.cpp
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.3
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "seres2netblock.h"

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
template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, int res2net_scale, int channel_se, typename T>
SERes2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_width, input_pad, res2net_scale, channel_se, T>::SERes2NetBlock()
{
    assert(input_width == out_width);

    // std::cout << "SERes2NetBlock initialized" << std::endl;
};

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
template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, int res2net_scale, int channel_se, typename T>
void SERes2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_width, input_pad, res2net_scale, channel_se, T>::forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_width])
{
    if (channel_in != channel_out)
    {
        this->shortcut.forward(input, this->residual);
    }
    else
    {
        MatrixFunctions::Copy(input, this->residual);
    }

    this->tdnn1.forward(input, this->x1);
    this->res2net.forward(this->x1, this->x1);
    this->tdnn2.forward(this->x1, this->x1);
    this->seblock.forward(this->x1, this->x1);

    // return x + residual
    MatrixFunctions::matrixAdd(this->x1, this->residual);
    MatrixFunctions::Copy(this->x1, output);
}

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
template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, int res2net_scale, int channel_se, typename T>
void SERes2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_width, input_pad, res2net_scale, channel_se, T>::forward(T (&input)[channel_in][input_width], T (&lengths)[channel_out], T (&output)[channel_out][out_width])
{
    if (channel_in != channel_out)
    {
        this->shortcut.forward(input, this->residual);
    }
    else
    {
        MatrixFunctions::Copy(input, this->residual);
    }

    this->tdnn1.forward(input, this->x1);
    this->res2net.forward(this->x1, this->x1);
    this->tdnn2.forward(this->x1, this->x1);
    this->seblock.forward(this->x1, lengths, this->x1);

    // return x + residual
    MatrixFunctions::matrixAdd(this->x1, this->residual);
    MatrixFunctions::Copy(this->x1, output);
}
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
template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, int res2net_scale, int channel_se, typename T>
void SERes2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_width, input_pad, res2net_scale, channel_se, T>::forward(T (&input)[channel_in][input_width], T &lengths, T (&output)[channel_out][out_width])
{
    if (channel_in != channel_out)
    {
        this->shortcut.forward(input, this->residual);
    }
    else
    {
        MatrixFunctions::Copy(input, this->residual);
    }

    this->tdnn1.forward(input, this->x1);
    this->res2net.forward(this->x1, this->x1);
    this->tdnn2.forward(this->x1, this->x1);
    this->seblock.forward(this->x1, lengths, this->x1);

    // return x + residual
    MatrixFunctions::matrixAdd(this->x1, this->residual);
    MatrixFunctions::Copy(this->x1, output);
}

template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, int res2net_scale, int channel_se, typename T>
SERes2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_width, input_pad, res2net_scale, channel_se, T>::~SERes2NetBlock(){};

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
template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, int res2net_scale, int channel_se, typename T>
void SERes2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_width, input_pad, res2net_scale, channel_se, T>::loadweights(std::string pathname)
{
    std::ifstream infile(pathname, std::ios::binary);
    if (!infile)
    {
        std::cout << "Error opening file!" << std::endl;
        return;
    }

    tdnn1.loadweights(infile);
    res2net.loadweights(infile);
    tdnn2.loadweights(infile);
    seblock.loadweights(infile);
    if (channel_in != channel_out)
    {
        shortcut.loadweights(infile);
    }

    infile.close();
};

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
template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, int res2net_scale, int channel_se, typename T>
void SERes2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_width, input_pad, res2net_scale, channel_se, T>::loadweights(std::ifstream &infile)
{
    // std::cout << 0 << std::endl;
    tdnn1.loadweights(infile);
    // std::cout << 1 << std::endl;
    res2net.loadweights(infile);
    // std::cout << 2 << std::endl;
    tdnn2.loadweights(infile);
    // std::cout << 3 << std::endl;
    seblock.loadweights(infile);
    // std::cout << 4 << std::endl;
    if (channel_in != channel_out)
    {
        shortcut.loadweights(infile);
        // std::cout << 5 << std::endl;
    }
};
