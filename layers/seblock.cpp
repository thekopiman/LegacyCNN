#include "seblock.h"

template <int channel_in, int channel_se, int channel_out, int input_width, int out_dim, typename T>
SEBlock<channel_in, channel_se, channel_out, input_width, out_dim, T>::SEBlock() : layer0(3), layer1(3)
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
    layer0.forward(mean, temp);
    ActivationFunctions::ReLU(temp);
    // We reuse mean to save space
    layer1.forward(temp, mean);
    ActivationFunctions::Sigmoid(mean);
    // Reshape mean[channel_in][1] to temp2[channel_in]
    MatrixFunctions::Reshape(mean, temp2);
    MatrixFunctions::HadamardProduct(input, temp2, output);
}

template <int channel_in, int channel_se, int channel_out, int input_width, int out_dim, typename T>
SEBlock<channel_in, channel_se, channel_out, input_width, out_dim, T>::~SEBlock() {}

template <int channel_in, int channel_se, int channel_out, int input_width, int out_dim, typename T>
void SEBlock<channel_in, channel_se, channel_out, input_width, out_dim, T>::loadweights(std::string pathname)
{
    std::ifstream infile(pathname, std::ios::binary);
    if (!infile)
    {
        std::cout << "Error opening file!" << std::endl;
        return;
    }
    layer0.loadweights(infile);
    layer1.loadweights(infile);
    infile.close();
}

template <int channel_in, int channel_se, int channel_out, int input_width, int out_dim, typename T>
void SEBlock<channel_in, channel_se, channel_out, input_width, out_dim, T>::loadweights(std::ifstream &infile)
{
    layer0.loadweights(infile);
    layer1.loadweights(infile);
}