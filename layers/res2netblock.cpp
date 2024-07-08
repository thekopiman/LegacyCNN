/**
 * @file res2netblock.cpp
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.1
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "res2netblock.h"

/**
 * @brief Construct a new Res2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_width, input_pad, scale, T>::Res2NetBlock object
 *
 * Make sure that input_width == out_width
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
Res2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_width, input_pad, scale, T>::Res2NetBlock()
{
    assert(channel_in % scale == 0);
    assert(channel_out % scale == 0);
    assert(input_width == out_width);

    // std::cout << "Res2NetBlock initialised" << std::endl;
};

template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, int scale, typename T>
Res2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_width, input_pad, scale, T>::~Res2NetBlock(){};

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
template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, int scale, typename T>
void Res2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_width, input_pad, scale, T>::forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_width])
{
    // std::cout << "Scale: " << scale << std::endl;
    // std::cout << "in_channel: " << channel_in / scale << std::endl;
    // std::cout << "input_width: " << input_width << std::endl;

    MatrixFunctions::Chunk(input, this->input_chunks);

    // std::cout << "Chunk ok " << std::endl;

    for (int i = 0; i < scale; i++)
    {
        if (i == 0)
        {
            MatrixFunctions::Copy(this->input_chunks[i], this->y_i);
        }
        else if (i == 1)
        {
            this->blocks[i - 1].forward(this->input_chunks[i], this->y_i);
        }
        else
        {
            resetTemp();
            // printTemp();
            MatrixFunctions::matrixAdd(this->temp, this->input_chunks[i]);
            // printTemp();
            MatrixFunctions::matrixAdd(this->temp, this->y_i);
            this->blocks[i - 1]
                .forward(this->temp, this->y_i);
            // printTemp();
        }

        // Append step
        MatrixFunctions::Copy(this->y_i, this->y[i]);
        // for (int j = 0; j < hidden_channel; j++)
        // {
        //     for (int k = 0; k < out_width; k++)
        //     {
        //         std::cout << this->y[i][j][k] << " ";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << std::endl;
    }
    MatrixFunctions::Cat(this->y, output);
};

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
template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, int scale, typename T>
void Res2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_width, input_pad, scale, T>::resetTemp()
{
    for (int i = 0; i < channel_out / scale; i++)
    {
        for (int j = 0; j < out_width; j++)
        {
            this->temp[i][j] = 0;
        }
    }
};
;

template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, int scale, typename T>
void Res2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_width, input_pad, scale, T>::printTemp()
{
    for (int i = 0; i < channel_out / scale; i++)
    {
        for (int j = 0; j < out_width; j++)
        {
            std::cout << this->y_i[i][j] << " ";
        }
        std::cout << std::endl;
    }
};

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
template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, int scale, typename T>
void Res2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_width, input_pad, scale, T>::loadweights(std::string pathname)
{
    std::ifstream infile(pathname, std::ios::binary);
    if (!infile)
    {
        std::cout << "Error opening file!" << std::endl;
        return;
    }

    for (int i = 0; i < scale - 1; i++)
    {
        // std::cout << "Current block: " << std::endl;
        blocks[i].loadweights(infile);
    }

    infile.close();
};

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
template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, int scale, typename T>
void Res2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_width, input_pad, scale, T>::loadweights(std::ifstream &infile)
{
    for (int i = 0; i < scale - 1; i++)
    {
        // std::cout << "Current block: " << std::endl;
        blocks[i].loadweights(infile);
    }
};

template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_width, int input_pad, int scale, typename T>
void Res2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_width, input_pad, scale, T>::printBlockParameters()
{
    for (int i = 0; i < scale - 1; i++)
    {
        std::cout << "Block " << i << std::endl;
        blocks[i].printparameters();
        std::cout << std::endl;
    }
}