#include "tdnnblock.h"
template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, typename T>
TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_dim, input_pad, T>::TDNNBlock() : layer0(1){
                                                                                                                    // std::cout << "TDNNBlock initialised" << std::endl;
                                                                                                                    // setWeights_layer0("ECAPAweights/test_weights.bin", false);
                                                                                                                    // setBias_layer0("ECAPAweights/test_bias.bin", false);
                                                                                                                };

template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, typename T>
TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_dim, input_pad, T>::~TDNNBlock(){};

template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, typename T>
void TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_dim, input_pad, T>::setWeights_layer0(T (&new_weights)[channel_out][channel_in][kernel])
{
    layer0.setWeights(new_weights);
};

template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, typename T>
void TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_dim, input_pad, T>::setBias_layer0(T (&new_bias)[channel_out])
{
    layer0.setBias(new_bias);
};

template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, typename T>
void TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_dim, input_pad, T>::setGamma_layer1(T (&new_gamma)[channel_out])
{
    layer1.setGamma(new_gamma);
};

template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, typename T>
void TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_dim, input_pad, T>::setBeta_layer1(T (&new_beta)[channel_out])
{
    layer1.setBeta(new_beta);
};

template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, typename T>
void TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_dim, input_pad, T>::setWeights_layer0(std::string pathname, bool displayWeights)
{
    layer0.setWeights(pathname, displayWeights);
};

template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, typename T>
void TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_dim, input_pad, T>::setBias_layer0(std::string pathname, bool displayBias)
{
    layer0.setBias(pathname, displayBias);
};

template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, typename T>
void TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_dim, input_pad, T>::setGamma_layer1(std::string pathname)
{
    layer1.setGamma(pathname);
};

template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, typename T>
void TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_dim, input_pad, T>::setBeta_layer1(std::string pathname)
{
    layer1.setBeta(pathname);
};

template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, typename T>
void TDNNBlock<kernel, stride, channel_in, channel_out, dilation, input_width, out_dim, input_pad, T>::forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim])
{
    layer0.forward(input, output);
    ActivationFunctions::ReLU<channel_out, out_dim, T>(output);
    layer1.forward(output, output);
};
