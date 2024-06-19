#include "seblock.h"

template <int channel_in, int channel_se, int channel_out, int input_width, int out_dim, typename T>
SEBlock<channel_in, channel_se, channel_out, input_width, out_dim, T>::SEBlock() : layer0(1), layer1(1)
{
    // Sanity check
    assert(input_width == out_dim);
    assert(channel_in == channel_out);

    // std::cout << "SEBlock is initialised" << std::endl;
}

template <int channel_in, int channel_se, int channel_out, int input_width, int out_dim, typename T>
void SEBlock<channel_in, channel_se, channel_out, input_width, out_dim, T>::forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim])
{
    MatrixFunctions::Mean(input, mean);
    layer0.forward(input, temp);
    ActivationFunctions::ReLU(temp);
    layer1.forward(temp, output);
    ActivationFunctions::Sigmoid(output);
    MatrixFunctions::HadamardProduct(output, mean, output);
}

template <int channel_in, int channel_se, int channel_out, int input_width, int out_dim, typename T>
SEBlock<channel_in, channel_se, channel_out, input_width, out_dim, T>::~SEBlock() {}