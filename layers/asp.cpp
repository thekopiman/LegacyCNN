#include "asp.h"

template <int channels, int attention_channels, int input_width, int out_dim, typename T>
ASP<channels, attention_channels, input_width, out_dim, T>::ASP(){};

template <int channels, int attention_channels, int input_width, int out_dim, typename T>
void ASP<channels, attention_channels, input_width, out_dim, T>::forward(T (&input)[channels][input_width], T (&output)[channels * 2][1])
{
    MatrixFunctions::Mean(input, this->mean);
    MatrixFunctions::Std(input, this->std);

    // attn = torch.cat([x, mean, std], dim=1)
    for (int i = 0; i < channels; i++)
    {
        for (int j = 0; j < input_width; j++)
        {
            this->attn[i][j] = input[i][j];
            this->attn[channels + i][j] = this->mean[i];    // mean = mean.unsqueeze(2).repeat(1, 1, L)
            this->attn[2 * channels + i][j] = this->std[i]; // std = std.unsqueeze(2).repeat(1, 1, L)
        }
    }

    // apply layers
    this->tdnn.forward(this->attn, this->attn1);
    ActivationFunctions::Tanh(this->attn1);
    this->conv.forward(this->attn1, this->attn2);

    // Filter out zero-paddings is redundant here

    ActivationFunctions::Softmax(this->attn2);

    // Obtain Pooled statistics
    compute_statistics(input, attn2);

    for (int i = 0; i < channels; i++)
    {
        output[i][0] = this->mean[i];
        output[channels + i][0] = this->std[i];
    }
};

template <int channels, int attention_channels, int input_width, int out_dim, typename T>
void ASP<channels, attention_channels, input_width, out_dim, T>::compute_statistics(T (&x)[channels][input_width], T (&m)[channels][input_width])
{
    T temp[channels][input_width];

    // Obtain Mean here
    MatrixFunctions::HadamardProduct(x, m, temp);
    for (int i = 0; i < channels; i++)
    {
        this->mean[i] = MatrixFunctions::Sum(temp[i]);
    }

    // Obtain Population Std here

    for (int i = 0; i < channels; i++)
    {
        for (int j = 0; j < input_width; j++)
        {
            x[i][j] -= this->mean[i]; // x - mean
            x[i][j] *= x[i][j];       // pow(2)
        }
    }

    MatrixFunctions::HadamardProduct(x, m, temp);
    for (int i = 0; i < channels; i++)
    {
        this->std[i] = MatrixFunctions::Clamp(MatrixFunctions::Sum(temp[i]), (T)1e-12);
        this->std[i] = std::sqrt(this->std[i]);
    }
}