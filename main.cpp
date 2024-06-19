#include <iostream>
#include "models/basiccnn.h"
#include "utils/helper.h"
#include "utils/activationfunctions.h"
#include "layers/res2netblock.h"
#include "layers/tdnnblock.h"
#include "models/ecapa.h"
int main()
{
    // The layers are statically allocated
    // Hence all values have to be given. ie. Precomputed beforehand
    std::cout << "Running program" << std::endl;
    // BasicCNNModel model;
    // Res2NetBlock<3, 8, 8, 1, 64, 64, 2, 8, float> model;
    // TDNNBlock<3, 1, 1, 1, 1, 64, 64, 2, float> model[5];
    ECAPA_TDNN model;
    float input[2][64];
    // float y[8][64];
    float y[6];

    // Helper::readInputs("BasicModelWeights/input.bin", input);
    Helper::readInputs("ECAPAweights/inputseres2_2.bin", input);
    // Helper::readInputs("ECAPAweights/inputseres2_1.bin", input);
    // model.loadweights();

    model.forward(input, y);

    for (int i = 0; i < 6; i++)
    {
        std::cout << y[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
