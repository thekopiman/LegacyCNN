#ifndef res2netblock_h
#define res2netblock_h

#include "conv1d.h"
#include "../utils/helper.h"
#include "tdnnblock.h"
#include <string>
#include <assert.h>

template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int scale, typename T>
class Res2NetBlock
{
public:
    Res2NetBlock()
    {
        assert(in_channels % scale == 0);
        assert(channel_out % scale == 0);
        std::cout << "Res2NetBlock initialised" << std::endl;
    }
    void getOutput(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim])
    {
    }
    ~Res2NetBlock()
    {
        delete (&blocks);
    }

private:
    TDNNBlock<kernel, 1, channel_in, channel_out / scale, 0, dilation, input_width, out_dim, T> blocks[scale - 1];
    int in_channel = channel_in / scale;
    int hidden_channel = channel_out / scale;
};

#endif res2netblock_h