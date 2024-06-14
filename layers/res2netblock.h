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
        assert(channel_in % scale == 0);
        assert(channel_out % scale == 0);

        std::cout << "Res2NetBlock initialised" << std::endl;
    }
    void forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim])
    {
        // std::cout << "Scale: " << scale << std::endl;
        // std::cout << "in_channel: " << channel_in / scale << std::endl;
        // std::cout << "input_width: " << input_width << std::endl;

        MatrixFunctions::Chunk(input, this->input_chunks);

        for (int i = 0; i < scale; i++)
        {
            if (i == 0)
            {
                MatrixFunctions::Copy(this->input_chunks[i], this->y_i);
            }
            else if (i == 1)
            {
                this->blocks[i - 1].forward(this->input_chunks[i], this->y_i);
            }
            else
            {
                resetTemp();

                // printTemp();
                MatrixFunctions::matrixAdd(this->temp, this->input_chunks[i]);
                // printTemp();
                MatrixFunctions::matrixAdd(this->temp, this->y_i);
                this->blocks[i - 1]
                    .forward(this->temp, this->y_i);
                // printTemp();
            }

            // Append step
            MatrixFunctions::Copy(this->y_i, this->y[i]);
        }
        MatrixFunctions::Cat(this->y, output);
    }

    void resetTemp()
    {
        for (int i = 0; i < channel_out / scale; i++)
        {
            for (int j = 0; j < out_dim; j++)
            {
                this->temp[i][j] = 0;
            }
        }
    };
    void printTemp()
    {
        for (int i = 0; i < channel_out / scale; i++)
        {
            for (int j = 0; j < out_dim; j++)
            {
                std::cout << this->y_i[i][j] << " ";
            }
            std::cout << std::endl;
        }
    };
    ~Res2NetBlock()
    {
    }

private:
    TDNNBlock<kernel, 1, channel_in / scale, channel_out / scale, dilation, input_width, out_dim, input_pad, T> blocks[scale - 1];
    int in_channel = channel_in / scale;
    int hidden_channel = channel_out / scale;
    T y_i[channel_out / scale][out_dim];
    T y[scale][channel_out / scale][out_dim];
    T temp[channel_out / scale][out_dim];
    T input_chunks[scale][channel_in / scale][input_width];
};

#endif