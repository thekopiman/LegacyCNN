#ifndef res2netblock_h
#define res2netblock_h

#include "conv1d.h"
#include "../utils/helper.h"
#include "../utils/matrixfunctions.h"
#include "tdnnblock.h"
#include <string>
#include <assert.h>

// kernel, channel_in, channel_out, dilation, input_width, out_dim, input_pad, scale, T
template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, int scale, typename T>
class Res2NetBlock
{
public:
    Res2NetBlock();
    void forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim]);
    void resetTemp();
    void printTemp();
    ~Res2NetBlock();
    void loadweights(std::string pathname);
    void loadweights(std::ifstream &infile);
    void printBlockParameters();

private:
    TDNNBlock<kernel, 1, channel_in / scale, channel_out / scale, dilation, input_width, out_dim, input_pad, T> blocks[scale - 1];
    int in_channel = channel_in / scale;
    int hidden_channel = channel_out / scale;
    T y_i[channel_out / scale][out_dim];
    T y[scale][channel_out / scale][out_dim];
    T temp[channel_out / scale][out_dim];
    T input_chunks[scale][channel_in / scale][input_width];
};

#include "res2netblock.cpp"

#endif