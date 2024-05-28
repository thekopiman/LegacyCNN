#include <iostream>
#include "conv1d.h" // Template Class

int main()
{
    // This layer is statically allocated
    // Hence all values have to be given. ie. Precomputed beforehand
    // Refer to Documentation to generate the values using python.

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

    for (int channel = 0; channel < 3; channel++)
    {
        for (int i = 0; i < 2; i++)
        {
            std::cout << output[channel][i] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}