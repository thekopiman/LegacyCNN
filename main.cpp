#include <iostream>
#include "models/basiccnn.h"
#include "utils/helper.h"
#include "utils/activationfunctions.h"
#include "layers/res2netblock.h"
#include "layers/tdnnblock.h"
#include "layers/seres2netblock.h"
#include "layers/res2netblock.h"
#include "layers/seblock.h"
#include "layers/asp.h"
#include "models/ecapa.h"

#include <fstream>
int main()
{
    // The layers are statically allocated
    // Hence all values have to be given. ie. Precomputed beforehand
    std::cout << "Running program" << std::endl;
    // Res2NetBlock<3, 8, 8, 1, 64, 64, 2, 8, float> model;
    // SERes2NetBlock<3, 8, 8, 2, 64, 64, 4, 8, 128, float> seres_1;
    // Res2NetBlock<3, 8, 8, 2, 64, 64, 4, 8, float> res2net;
    // SEBlock<8, 128, 8, 64, 64, float> seblock;
    // ASP<16, 128, 64, 2, float> asp;

    ECAPA_TDNN model;
    float input[2][64];
    // float y[8][64];
    // float y[32][1];
    float y[6];

    // Helper::readInputs("ECAPAweights/aspinput.bin", input);
    // Helper::readInputs("ECAPAweights/seresinput.bin", input);
    Helper::readInputs("ECAPAweights/ecapainput.bin", input);

    // seres_1.loadweights("ECAPAweights/seres2_1.bin");
    // seres_1.forward(input, y);

    model.loadweights("ECAPAweights/fullecapa.bin");
    model.forward(input, y);

    // asp.loadweights("ECAPAweights/asp.bin");
    // asp.forward(input, y);

    ActivationFunctions::Softmax(y);
    Helper::print(y);

    return 0;
}
