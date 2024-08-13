#include <iostream>
#include "tests/time.h"
#include "utils/matrixfunctions.h"
#include "utils/helper.h"
#include "models/ecapa_classifier.h"
#include "layers/seblock.h"
#include "layers/seres2netblock.h"

int main()
{
    // Time::TestEcapa(100);
    // Time::TestBasic(100);

    std::cout << "Testing Ecapa" << std::endl;

    // Initialise model
    // ECAPA_TDNN ecapamodel;

    // Cosine = 0, CDist = 1, Euclidean = 2
    ECAPA_TDNN_classifier ecapamodel(0);
    float y[6][6];

    // Load weights
    ecapamodel.loadweights("ECAPAweights/fullecapa_classifier.bin");

    // Metrics on Read Input will not be tested
    float input[2][64];
    Helper::readInputs("ECAPAweights/ecapainput_2x64.bin", input);

    float lengths = 0.4;

    ecapamodel.forward(input, lengths, y);
    Helper::print(y);

    return 0;
}
