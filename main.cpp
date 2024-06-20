#include <iostream>
#include "models/basiccnn.h"
#include "utils/helper.h"
#include "utils/activationfunctions.h"
#include "layers/res2netblock.h"
#include "layers/tdnnblock.h"
#include "layers/seres2netblock.h"
#include "layers/res2netblock.h"
#include "layers/seblock.h"
#include "models/ecapa.h"

#include <fstream>
int main()
{
    // The layers are statically allocated
    // Hence all values have to be given. ie. Precomputed beforehand
    std::cout << "Running program" << std::endl;
    // Res2NetBlock<3, 8, 8, 1, 64, 64, 2, 8, float> model;
    SERes2NetBlock<3, 8, 8, 2, 64, 64, 4, 8, 128, float> seres_1;
    // Res2NetBlock<3, 8, 8, 2, 64, 64, 4, 8, float> res2net;
    // SEBlock<8, 128, 8, 64, 64, float> seblock;

    ECAPA_TDNN model;
    float input[2][64];
    // float y[8][64];
    float y[6];

    Helper::readInputs("ECAPAweights/ecapainput.bin", input);
    // Helper::readInputs("ECAPAweights/seresinput.bin", input);

    // seres_1.loadweights("ECAPAweights/seres2_1.bin");
    // seres_1.forward(input, y);

    model.loadweights("ECAPAweights/fullecapa.bin");
    model.forward(input, y);

    // for (int i = 0; i < 8; i++)
    // {
    //     for (int j = 0; j < 64; j++)
    //     {
    //         // std::cout << input[i][j] << "(" << y[i][j] << ") ";
    //         // std::cout << input[i][j] << " ";
    //         std::cout << y[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    for (int i = 0; i < 6; i++)
    {
        std::cout << y[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
