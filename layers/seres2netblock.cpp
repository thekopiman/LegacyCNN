#include "seres2netblock.h"

template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, int res2net_scale, int channel_se, typename T>
SERes2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_dim, input_pad, res2net_scale, channel_se, T>::SERes2NetBlock()
{
    assert(input_width == out_dim);

    // std::cout << "SERes2NetBlock initialized" << std::endl;
};

template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, int res2net_scale, int channel_se, typename T>
void SERes2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_dim, input_pad, res2net_scale, channel_se, T>::forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim])
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
    this->tdnn1.forward(this->x1, this->x1);
    this->seblock.forward(this->x1, this->x1);

    // return x + residual
    MatrixFunctions::matrixAdd(this->x1, this->residual);
    MatrixFunctions::Copy(this->x1, output);
}

template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, int res2net_scale, int channel_se, typename T>
SERes2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_dim, input_pad, res2net_scale, channel_se, T>::~SERes2NetBlock(){};
