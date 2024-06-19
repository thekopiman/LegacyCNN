#ifndef asp_h
#define asp_h

#include "conv1d.h"
#include "../utils/helper.h"
#include "../utils/matrixfunctions.h"
#include "../utils/activationfunctions.h"
#include "tdnnblock.h"
#include <string>
#include <assert.h>
#include <limits>

// We assume global_context = True and Lengths = None
template <int channels, int attention_channels, int input_width, int out_dim, typename T>
class ASP
{
public:
    ASP()
    {
    }

    void forward(T (&input)[channels][input_width], T (&output)[channels * 2][1])
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
    }

private:
    TDNNBlock<1, 1, channels * 3, attention_channels, 1, input_width, input_width, 0, T> tdnn;
    Conv1d<1, 1, attention_channels, channels, 0, 1, input_width, input_width, T> conv;
    T attn[channels * 3][input_width];
    T attn1[attention_channels][input_width];
    T attn2[channels][input_width];
    T mean[channels];
    T std[channels];

    void compute_statistics(T (&x)[channels][input_width], T (&m)[channels][input_width])
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
};

#endif