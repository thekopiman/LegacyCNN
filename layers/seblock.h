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
template <int kernel, int channel_in, int channel_se, int channel_out, int input_width, int out_dim, int input_pad, typename T>
class SEBlock
{
public:
    SEBlock() : layer0(1), layer1(1)
    {
        // Sanity check
        assert(input_width == out_dim);
        assert(channel_in == channel_out);

        std::cout << "SEBlock is initialised" << std::endl;
    }
    void forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim])
    {
        MatrixFunctions::Mean(input, mean);
        layer0.forward(input, output);
        ActivationFunctions::ReLU(output);
        layer1.forward(output, output);
        ActivationFunctions::Sigmoid(output);
        MatrixFunctions::HadamardProduct(output, mean, output);
    }

private:
    Conv1d<1, 1, channel_in, channel_se, 0, 1, input_width, out_dim, T> layer0;
    Conv1d<1, 1, channel_se, channel_out, 0, 1, out_dim, out_dim, T> layer1;
    T mean[channel_in];
};

#endif
