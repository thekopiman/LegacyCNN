#include "tdnnblock.h"
#include <fstream>

template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, typename T>
TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_dim, input_pad, T>::TDNNBlock() : layer0(3), layer1(1){};

template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, typename T>
TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_dim, input_pad, T>::~TDNNBlock(){};

template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, typename T>
void TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_dim, input_pad, T>::forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim])
{
    layer0.forward(input, output);
    ActivationFunctions::ReLU<channel_out, out_dim, T>(output);
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

template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, typename T>
void TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_dim, input_pad, T>::loadweights(std::string pathname)
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

template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, typename T>
void TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_dim, input_pad, T>::loadweights(std::ifstream &infile)
{
    if (!infile)
    {
        std::cout << "Error opening file!" << std::endl;
        return;
    }
    layer0.loadweights(infile);
    layer1.loadweights(infile);
};

template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, typename T>
void TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_dim, input_pad, T>::printparameters()
{
    std::cout << "kernel " << kernel << std::endl;
    std::cout << "stride " << stride << std::endl;
    std::cout << "channel_in " << channel_in << std::endl;
    std::cout << "channel_out " << channel_out << std::endl;
    std::cout << "dilation " << dilation << std::endl;
    std::cout << "input_pad " << input_pad << std::endl;
    // std::cout << "Kernel " << kernel << std::endl;
}