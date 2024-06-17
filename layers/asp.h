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

    void forward(T (&input)[channels][input_width], T (&output)[2][channels])
    {
        MatrixFunctions::Mean(input, this->mean);
        MatrixFunctions::Std(input, this->std);

        // attn = torch.cat([x, mean, std], dim=1)
        for (int i = 0; i < channels; i++)
        {
            for (int j = 0; j < input_width; i++)
            {
                this->attn[i][j] = this->input[i][j];
                this->attn[channels + i][j] = this->mean[i];
                this->attn[2 * channels + i][j] = this->std[i];
            }
        }

        // apply layers
        this->tdnn.forward(this->attn, this->attn1);
        ActivationFunctions::Tanh(this->attn1);
        this->conv.forward(this->attn1, this->attn2);

        // Filter out zero-paddings
        for (int i = 0; i < channels; i++)
        {
            for (int j = 0; j < input_width; j++)
            {
                if (this->attn2[i][j] == 0)
                {
                    this->attn2[i][j] = std::numeric_limits<T>::lowest();
                }
            }
        }

        ActivationFunctions::Softmax(this->attn2);

        // Obtain Pooled statistics
        MatrixFunctions::Mean(MatrixFunctions::HadamardProduct(this->input, this->attn2), this->mean);
        MatrixFunctions::Std(MatrixFunctions::HadamardProduct(this->input, this->attn2), this->std);

        for (int i = 0; i < channels; i++)
        {
            this->output[0][i] = this->mean[i];
            this->output[1][i] = this->std[i];
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
};

#endif