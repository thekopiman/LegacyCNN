#ifndef seblock_h
#define seblock_h

#include "conv1d.h"
#include "../utils/helper.h"
#include "../utils/activationfunctions.h"
#include "../utils/matrixfunctions.h"
#include "tdnnblock.h"
#include <string>
#include <assert.h>
#include <fstream>

// With ref to the python code, we always assume lengths = None
// channel_in, channel_se, channel_out, input_width, out_dim, T
// <8, 128, 8, 64, 64, float>
template <int channel_in, int channel_se, int channel_out, int input_width, int out_dim, typename T>
class SEBlock
{
public:
    SEBlock();
    void forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim]);
    ~SEBlock();
    void loadweights(std::string pathname);
    void loadweights(std::ifstream &infile);

private:
    Conv1d<1, 1, channel_in, channel_se, 0, 1, 1, 1, T> layer0;
    T temp[channel_se][1];
    Conv1d<1, 1, channel_se, channel_out, 0, 1, 1, 1, T> layer1;
    T mean[channel_in][1];
    T temp2[channel_out];
};

#include "seblock.cpp"

#endif
