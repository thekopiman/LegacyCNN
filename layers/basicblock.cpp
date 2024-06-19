#include "basicblock.h"

template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_dim, typename T>

BasicBlock<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_dim, T>::BasicBlock(){
    // std::cout << "BasicBlock initialised" << std::endl;
};

template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_dim, typename T>

BasicBlock<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_dim, T>::~BasicBlock(){};

template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_dim, typename T>

void BasicBlock<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_dim, T>::setWeights_layer0(T (&new_weights)[channel_out][channel_in][kernel])
{
    this->layer0.setWeights(new_weights);
};

template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_dim, typename T>

void BasicBlock<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_dim, T>::setBias_layer0(T (&new_bias)[channel_out])
{
    this->layer0.setBias(new_bias);
};

template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_dim, typename T>

void BasicBlock<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_dim, T>::setGamma_layer1(T (&new_gamma)[channel_out])
{
    this->layer1.setGamma(new_gamma);
}
template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_dim, typename T>

void BasicBlock<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_dim, T>::setBeta_layer1(T (&new_beta)[channel_out])
{
    this->layer1.setBeta(new_beta);
};

template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_dim, typename T>

void BasicBlock<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_dim, T>::setWeights_layer0(std::string pathname, bool displayWeights)
{
    this->layer0.setWeights(pathname, displayWeights);
};

template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_dim, typename T>

void BasicBlock<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_dim, T>::setBias_layer0(std::string pathname, bool displayBias)
{
    this->layer0.setBias(pathname, displayBias);
};

template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_dim, typename T>

void BasicBlock<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_dim, T>::setGamma_layer1(std::string pathname)
{
    this->layer1.setGamma(pathname);
};

template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_dim, typename T>

void BasicBlock<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_dim, T>::setBeta_layer1(std::string pathname)
{
    this->layer1.setBeta(pathname);
};

template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_dim, typename T>

void BasicBlock<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_dim, T>::forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim])
{
    this->layer0.forward(input, output);
    ActivationFunctions::ReLU<channel_out, out_dim, T>(output);
    this->layer1.forward(output, output);
}
