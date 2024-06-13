#include <iostream>
#include "models/basiccnn.h"
#include "utils/helper.h"
#include "utils/activationfunctions.h"

int main()
{
    // The layers are statically allocated
    // Hence all values have to be given. ie. Precomputed beforehand
    std::cout << "Running program" << std::endl;
    BasicCNNModel model;
    float input[2][16];
    float y[6];

    Helper::readInputs("BasicModelWeights/input.bin", input);
    model.loadweights();
    model.forward(input, y);

    for (int i = 0; i < 6; i++)
    {
        std::cout << y[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}