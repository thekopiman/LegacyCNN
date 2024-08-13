#include <iostream>
#include "tests/time.h"
#include "utils/matrixfunctions.h"
#include "utils/helper.h"
#include "models/ecapa_classifier.h"

int main()
{
    std::cout << "Running program" << std::endl;

    // Time::TestEcapa(100);

    // Time::TestBasic(100);

    std::cout << "Testing Ecapa" << std::endl;

    // Initialise model
    // ECAPA_TDNN ecapamodel;
    ECAPA_TDNN_classifier ecapamodel(false);
    float y[6][6];

    // Load weights
    ecapamodel.loadweights("ECAPAweights/fullecapa_classifier.bin");

    // Metrics on Read Input will not be tested
    // float input_full[100][2][64];
    float input[2][64];
    std::cout << "Before Loading" << std::endl;
    Helper::readInputs("ECAPAweights/ecapainput_2x64.bin", input);
    std::cout << "Complete Loading" << std::endl;

    ecapamodel.forward(input, y);
    Helper::print(y);

    return 0;
}
