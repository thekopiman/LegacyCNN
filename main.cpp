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
    TDNNBlock<5, 1, 2, 8, 1, 64, 64, 4, float> model;
    // ECAPA_TDNN model;
    float input[2][64];
    float y[8][64];
    // float y[6];

    // Helper::readInputs("BasicModelWeights/input.bin", input);
    Helper::readInputs("ECAPAweights/initialinput.bin", input);
    // Helper::readInputs("ECAPAweights/inputseres2_1.bin", input);

    model.setWeights_full("ECAPAweights/initialblock.bin");

    model.forward(input, y);

    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 64; j++)
        {
            // std::cout << input[i][j] << "(" << y[i][j] << ") ";
            std::cout << y[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    return 0;
}
