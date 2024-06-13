#ifndef res2netblock_h
#define res2netblock_h

#include "conv1d.h"
#include "../utils/helper.h"
#include "../utils/matrixfunctions.h"
#include "tdnnblock.h"
#include <string>
#include <assert.h>

template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, int scale, typename T>
class Res2NetBlock
{
public:
    Res2NetBlock()
    {
        assert(in_channels % scale == 0);
        assert(channel_out % scale == 0);

        // Initialize y_i
        for (int i = 0; i < out_dim; i++)
        {
            this->y_i[i] = 0;
        }

        std::cout << "Res2NetBlock initialised" << std::endl;
    }
    void getOutput(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim])
    {
        for (int i = 0; i < scale; i++)
        {
            if (i == 0)
            {
                this->y_i = input[i];
            }
            else if (i == 1)
            {
                this->blocks[i - 1].getOutput(input[i], this->y_i);
            }
            else
            {
                resetTemp();
                MatrixFunctions::Sum(this->temp, input[i]);
                MatrixFunctions::Sum(this->temp, this->y_i);
                this->blocks[i - 1]
                    .getOutput(this->temp, this->y_i);
            }

            // output[i] = this->y_i

            for (int j = 0; j < out_dim; j++)
            {
                output[i][j] = this->y_i[j];
            }
        }
    }

    void resetTemp()
    {
        for (int i = 0; i < out_dim; i++)
        {
            this->temp[i] = 0;
        }
    }
    ~Res2NetBlock()
    {
    }

private:
    TDNNBlock<kernel, 1, channel_in, channel_out / scale, 0, dilation, input_width, out_dim, input_pad T> blocks[scale - 1];
    int in_channel = channel_in / scale;
    int hidden_channel = channel_out / scale;
    int y_i[out_dim];
    int temp[out_dim];
};

#endif res2netblock_h