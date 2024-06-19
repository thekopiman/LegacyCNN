#include "res2netblock.h"

template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, int scale, typename T>
Res2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_dim, input_pad, scale, T>::Res2NetBlock()
{
    assert(channel_in % scale == 0);
    assert(channel_out % scale == 0);

    // std::cout << "Res2NetBlock initialised" << std::endl;
};

template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, int scale, typename T>
Res2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_dim, input_pad, scale, T>::~Res2NetBlock(){};

template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, int scale, typename T>
void Res2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_dim, input_pad, scale, T>::forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim])
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
            // std::cout << "i = 0 ok " << std::endl;
        }
        else if (i == 1)
        {
            this->blocks[i - 1].forward(this->input_chunks[i], this->y_i);
            // std::cout << "i = 1 ok " << std::endl;
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
            // std::cout << "i = ？ ok " << std::endl;
        }

        // Append step
        MatrixFunctions::Copy(this->y_i, this->y[i]);
        // for (int j = 0; j < hidden_channel; j++)
        // {
        //     for (int k = 0; k < out_dim; k++)
        //     {
        //         std::cout << this->y[i][j][k] << " ";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << std::endl;
    }
    MatrixFunctions::Cat(this->y, output);
};

template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, int scale, typename T>
void Res2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_dim, input_pad, scale, T>::resetTemp()
{
    for (int i = 0; i < channel_out / scale; i++)
    {
        for (int j = 0; j < out_dim; j++)
        {
            this->temp[i][j] = 0;
        }
    }
};
;

template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, int scale, typename T>
void Res2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_dim, input_pad, scale, T>::printTemp()
{
    for (int i = 0; i < channel_out / scale; i++)
    {
        for (int j = 0; j < out_dim; j++)
        {
            std::cout << this->y_i[i][j] << " ";
        }
        std::cout << std::endl;
    }
};