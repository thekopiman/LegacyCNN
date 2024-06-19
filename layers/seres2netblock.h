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

// kernel, channel_in, channel_out, dilation, input_width, out_dim, input_pad, res2net_scale,channel_se, T
template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, int res2net_scale, int channel_se, typename T>
class SERes2NetBlock
{
public:
    SERes2NetBlock()
    {
        assert(input_width == out_dim);

        // std::cout << "SERes2NetBlock initialized" << std::endl;
    }
    void forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim])
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
        this->tdnn1.forward(this->x1, this->x1);
        this->seblock.forward(this->x1, this->x1);

        // return x + residual
        MatrixFunctions::matrixAdd(this->x1, this->residual);
        MatrixFunctions::Copy(this->x1, output);
    }

private:
    TDNNBlock<1, 1, channel_in, channel_out, 1, input_width, out_dim, 0, T> tdnn1;
    T x1[channel_out][out_dim];
    Res2NetBlock<kernel, channel_out, channel_out, dilation, input_width, out_dim, input_pad, res2net_scale, T> res2net;
    TDNNBlock<1, 1, channel_out, channel_out, 1, input_width, out_dim, 0, T> tdnn2;
    SEBlock<channel_out, channel_se, channel_out, input_width, out_dim, T> seblock;
    Conv1d<1, 1, channel_in, channel_out, 0, 1, input_width, out_dim, T> shortcut;
    T residual[channel_out][out_dim];
};

#endif
