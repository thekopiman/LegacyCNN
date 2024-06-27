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
    this->tdnn2.forward(this->x1, this->x1);
    this->seblock.forward(this->x1, this->x1);

    // return x + residual
    MatrixFunctions::matrixAdd(this->x1, this->residual);
    MatrixFunctions::Copy(this->x1, output);
}

template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, int res2net_scale, int channel_se, typename T>
SERes2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_dim, input_pad, res2net_scale, channel_se, T>::~SERes2NetBlock(){};

template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, int res2net_scale, int channel_se, typename T>
void SERes2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_dim, input_pad, res2net_scale, channel_se, T>::loadweights(std::string pathname)
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

template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, int res2net_scale, int channel_se, typename T>
void SERes2NetBlock<kernel, channel_in, channel_out, dilation, input_width, out_dim, input_pad, res2net_scale, channel_se, T>::loadweights(std::ifstream &infile)
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
