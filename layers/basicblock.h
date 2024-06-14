#ifndef basiccnn_h
#define basiccnn_h

#include "conv1d.h"
#include "../utils/helper.h"
#include "../utils/activationfunctions.h"
#include "../utils/matrixfunctions.h"
#include "batchnorm1d.h"
#include <string>

// Conv1d
// ReLU
// BatchNorm
template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_dim, typename T>
class BasicBlock
{
public:
    BasicBlock()
    {
        // std::cout << "BasicBlock initialised" << std::endl;
    }

    // Set weights directly
    void setWeights_layer0(T (&new_weights)[channel_out][channel_in][kernel])
    {
        layer0.setWeights(new_weights);
    };
    void setBias_layer0(T (&new_bias)[channel_out])
    {
        layer0.setBias(new_bias);
    };
    void setGamma_layer1(T (&new_gamma)[channel_out])
    {
        layer1.setGamma(new_gamma);
    };
    void setBeta_layer1(T (&new_beta)[channel_out])
    {
        layer1.setBeta(new_beta);
    };

    // Set weights from path via overloading
    void setWeights_layer0(std::string pathname, bool displayWeights)
    {
        layer0.setWeights(pathname, displayWeights);
    };
    void setBias_layer0(std::string pathname, bool displayBias)
    {
        layer0.setBias(pathname, displayBias);
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
    void forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim])
    {
        layer0.forward(input, output);
        ActivationFunctions::ReLU<channel_out, out_dim, T>(output);
        layer1.forward(output, output);
    }

    ~BasicBlock()
    {
    }

private:
    Conv1d<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_dim, T> layer0;
    BatchNorm1d<channel_out, out_dim, T> layer1;
};

#endif