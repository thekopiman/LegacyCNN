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
    BasicBlock();

    // Set weights directly
    void setWeights_layer0(T (&new_weights)[channel_out][channel_in][kernel]);
    void setBias_layer0(T (&new_bias)[channel_out]);
    void setGamma_layer1(T (&new_gamma)[channel_out]);
    void setBeta_layer1(T (&new_beta)[channel_out]);

    // Set weights from path via overloading
    void setWeights_layer0(std::string pathname, bool displayWeights);
    void setBias_layer0(std::string pathname, bool displayBias);
    void setGamma_layer1(std::string pathname);
    void setBeta_layer1(std::string pathname);

    // Get Output
    void forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim]);

    ~BasicBlock();

private:
    Conv1d<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_dim, T> layer0;
    BatchNorm1d<channel_out, out_dim, T> layer1;
};

#include "basicblock.cpp"

#endif