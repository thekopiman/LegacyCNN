#ifndef basiccnn_h
#define basiccnn_h

#include "conv1d.h"
#include "helper.h"
#include "batchnorm1d.h"
#include <string>

// Conv1d
// ReLU
// BatchNorm
template <int kernal, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_dim>
class BasicCNNBlock
{
public:
    // Set weights directly
    void setWeights_layer0(float (&new_weights)[channel_out][channel_in][kernal])
    {
        layer0.setWeights(new_weights);
    };
    void setBias_layer0(float (&new_bias)[channel_out])
    {
        layer0.setBias(new_bias);
    };
    void setGamma_layer1(float (&new_gamma)[channel_out])
    {
        layer1.setGamma(new_gamma);
    };
    void setBeta_layer1(float (&new_beta)[channel_out])
    {
        layer1.setBeta(new_beta);
    };

    // Set weights from path via overloading
    void setWeights_layer0(std::string pathname)
    {
        layer0.setWeights(pathname);
    };
    void setBias_layer0(std::string pathname)
    {
        layer0.setBias(pathname);
    };
    void setGamma_layer1(std::string pathname)
    {
        layer1.setGamma(pathname);
    };
    void setBeta_layer1(std::string pathname)
    {
        layer1.setBeta(pathname);
    };

    // Get Output
    void getOutput(float (&input)[channel_in][input_width], float (&output)[channel_out][out_dim])
    {
        layer0.getOutput(input, output);
        Helper::ReLU<channel_out, out_dim>(output);
        layer1.getOutput(output, output);
    }

private:
    Conv1d<kernal, stride, channel_in, channel_out, pad, dilation, input_width, out_dim> layer0;
    BatchNorm1d<channel_out, out_dim> layer1;
};

#endif