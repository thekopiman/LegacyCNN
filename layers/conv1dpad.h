#ifndef conv1dpad_h
#define conv1dpad_h

#include <fstream>
#include <assert.h>
#include <string>
#include <iostream>
#include "conv1d.h"

// This Conv1dPad aims to replicate the autopadding conv1d used in
// https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/nnet/CNN.html#Conv1d
//
// input_pad = dilation * (kernel - 1) if stride = 1
// Else use the following formula
// input_width == ((input_width + input_pad - dilation * (kernel - 1) - 1) / stride) + 1
// to determine input_pad
//
// If you intend to utilize autopadding where input dimension and output dimension are equal
// Here we assume pad = 0
template <int kernel, int stride, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, typename T>
class Conv1dPad
{
public:
    Conv1dPad()
    {
        // Sanity Check
        assert(input_width == out_dim && "Ensure that Input and output dim are indentical");

        assert(input_width == ((input_width + input_pad - dilation * (kernel - 1) - 1) / stride) + 1 && "Ensure that input_pad is correct")
    };

    void setWeights(T (&new_weights)[channel_out][channel_in][kernel])
    {
        NormalLayer.setWeights(new_weights);
    };
    void setWeights(T (&new_weights)[channel_out * channel_in * kernel])
    {
        NormalLayer.setWeights(new_weights);
    };

    void setBias(T (&new_bias)[channel_out]){
        NormalLayer.setBias(new_bias)};

    void setWeights(std::string filename, bool displayWeights)
    {
        NormalLayer.setWeights(filename, displayWeights);
    };
    void setBias(std::string filename, bool displayBias)
    {
        NormalLayer.setBias(filename, displayWeights);
    };
    void getOutput(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim])
    {

        for (int i = 0; i < channel_in; i++)
        {
            for (int j = 0; j < input_width + input_pad; j++)
            {
                if (j < input_pad)
                {
                    this->new_input[i][j] = 0;
                }
                else
                {
                    this->new_input[i][j] = input[i][j - input_pad];
                }
            }
        }

        NormalLayer.getOutput(this->new_input, output);
    };

private:
    Conv1d<kernel, stride, channel_in, channel_out, 0, dilation, input_pad + input_pad, out_dim, T> NormalLayer;
    T new_input[channel_in][input_width + input_pad];
};
#endif