#ifndef seres2netblock_h
#define seres2netblock_h

#include "conv1d.h"
#include "../utils/helper.h"
#include "../utils/matrixfunctions.h"
#include "tdnnblock.h"
#include "res2netblock.h"
#include "seblock.h"
#include <string>
#include <assert.h>
#include <fstream>

// kernel, channel_in, channel_out, dilation, input_width, out_dim, input_pad, res2net_scale,channel_se, T
template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, int res2net_scale, int channel_se, typename T>
class SERes2NetBlock
{
public:
    SERes2NetBlock();
    void forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim]);
    ~SERes2NetBlock();
    void loadweights(std::string pathname);
    void loadweights(std::ifstream &infile);

private:
    TDNNBlock<1, 1, channel_in, channel_out, 1, input_width, out_dim, 0, T> tdnn1;
    T x1[channel_out][out_dim];
    Res2NetBlock<kernel, channel_out, channel_out, dilation, input_width, out_dim, input_pad, res2net_scale, T> res2net;
    TDNNBlock<1, 1, channel_out, channel_out, 1, input_width, out_dim, 0, T> tdnn2;
    SEBlock<channel_out, channel_se, channel_out, input_width, out_dim, T> seblock;
    Conv1d<1, 1, channel_in, channel_out, 0, 1, input_width, out_dim, T> shortcut;
    T residual[channel_out][out_dim];
};

#include "seres2netblock.cpp"

#endif
