#include <iostream>
#include "layers/conv1d.h"
#include "layers/dense.h"
#include "layers/helper.h"

int main()
{
    // This layer is statically allocated
    // Hence all values have to be given. ie. Precomputed beforehand
    // Refer to Documentation to generate the values using python.

    // Kernal, Stride, Channel_in, Channel_out, pad, dilation, input_width, out_dim

    Conv1d<3, 1, 2, 3, 0, 2, 6, 2> layer1;
    float weights[3][2][3] = {{{0, 0, 0}, {0, 0, 0}},
                              {{1, 1, 1}, {1, 1, 1}},
                              {{2, 2, 2}, {2, 2, 2}}};
    float bias[3] = {0, 1, 2};

    layer1.setWeights(weights);
    layer1.setBias(bias);

    float input[2][6] = {{1, 2, 3, 1, 2, 3}, {4, 5, 6, 4, 5, 6}};
    float output[3][2];
    layer1.getOutput(input, output);

    Helper::Softmax(output);

    float flatten_output[3 * 2];

    Helper::Flatten(output, flatten_output);
    for (int i = 0; i < 6; i++)
    {
        std::cout << flatten_output[i] << " ";
    }
    std::cout << std::endl;

    // FC Layer
    Dense<6, 2> FClayer;
    float FC_weights[6][2] = {{0, 1},
                              {0, 1},
                              {0, 1},
                              {0, 1},
                              {0, 1},
                              {0, 1}};
    float FC_bias[2] = {2, 5};
    FClayer.setWeights(FC_weights);
    FClayer.setBias(FC_bias);

    float y[2];
    FClayer.getOutput(flatten_output, y);

    for (int i = 0; i < 2; i++)
    {
        std::cout << y[i] << " ";
    }

    return 0;
}