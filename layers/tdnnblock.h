#ifndef tdnnblock_h
#define tdnnblock_h

#include "conv1d.h"
#include "../utils/helper.h"
#include "../utils/activationfunctions.h"
#include "../utils/matrixfunctions.h"
#include "batchnorm1d.h"
#include <string>
#include <fstream>

// kernel, stride, channel_in, channel_out, dilation, input_width, out_dim, input_pad, T
//
// Layers:
// Conv1d
// ReLU
// BatchNorm
//
// We assume pad = 0 (excluding input_pad)
// Make sure you calculate input_pad properly so that input_width == out_dim
template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, typename T>
class TDNNBlock
{
public:
    TDNNBlock();

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

    // Set weights from infile via overloading
    void setWeights_layer0(std::ifstream &infile, bool displayWeights);
    void setBias_layer0(std::ifstream &infile, bool displayBias);
    void setGamma_layer1(std::ifstream &infile);
    void setBeta_layer1(std::ifstream &infile);

    void setWeights_full(std::string pathname);
    void setWeights_full(std::ifstream &infile);

    // Get Output
    void forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim]);

    // Print Block parameters
    void printparameters();

    ~TDNNBlock();

private:
    Conv1d<kernel, stride, channel_in, channel_out, input_pad, dilation, input_width, out_dim, T> layer0;
    BatchNorm1d<channel_out, out_dim, T> layer1;
};

#include "tdnnblock.cpp"

#endif