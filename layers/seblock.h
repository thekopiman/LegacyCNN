#ifndef seblock_h
#define seblock_h

#include "conv1d.h"
#include "../utils/helper.h"
#include "../utils/activationfunctions.h"
#include "../utils/matrixfunctions.h"
#include "tdnnblock.h"
#include <string>
#include <assert.h>

// With ref to the python code, we always assume lengths = None
template <int channel_in, int channel_se, int channel_out, int input_width, int out_dim, typename T>
class SEBlock
{
public:
    SEBlock();
    void forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim]);
    ~SEBlock();

private:
    Conv1d<1, 1, channel_in, channel_se, 0, 1, input_width, out_dim, T> layer0;
    T temp[channel_se][input_width];
    Conv1d<1, 1, channel_se, channel_out, 0, 1, out_dim, out_dim, T> layer1;
    T mean[channel_in];
};

#include "seblock.cpp"

#endif
