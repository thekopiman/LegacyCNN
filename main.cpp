#include <iostream>
#include "tests/time.h"
#include "utils/matrixfunctions.h"
#include "utils/helper.h"

int main()
{
    std::cout << "Running program" << std::endl;

    // Time::TestEcapa(100);

    // Time::TestBasic(100);

    std::cout << "Testing Ecapa" << std::endl;

    // Initialise model
    ECAPA_TDNN ecapamodel;
    float y[6];

    // Load weights
    ecapamodel.loadweights("ECAPAweights/fullecapa.bin");

    // Metrics on Read Input will not be tested
    float input_full[10][2][64];
    float input[2][64];
    std::cout << "Before Loading" << std::endl;
    Helper::readInputs("ECAPAweights/ecapainput_2x64_10.bin", input_full);
    std::cout << "Complete Loading" << std::endl;

    for (int i = 0; i < 10; i++)
    {
        MatrixFunctions::Copy(input_full[i], input);
        ecapamodel.forward(input, y);
        Helper::print(y);
    }

    return 0;
}
