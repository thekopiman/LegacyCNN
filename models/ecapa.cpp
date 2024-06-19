#include "ecapa.h"

void ECAPA_TDNN::forward(float (&input)[2][64], float (&y)[6])
{
    initiallayer.forward(input, x0);

    seres_1.forward(x0, x1);

    seres_2.forward(x1, x2);

    seres_3.forward(x2, x3);

    // Cat
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 64; j++)
        {
            x_cat[i][j] = x1[i][j];
            x_cat[8 + i][j] = x2[i][j];
            x_cat[16 + i][j] = x3[i][j];
        }
    }

    mfa.forward(x_cat, y0);

    // ASP has errors
    asp.forward(y0, y1);

    asp_BN.forward(y1, y1);

    MatrixFunctions::Flatten(y1, flatten_x);

    fc.forward(flatten_x, y);
};

ECAPA_TDNN::ECAPA_TDNN()
{
    std::cout << "ECAPA TDNN initialised" << std::endl;
};

ECAPA_TDNN::~ECAPA_TDNN() {
};

void ECAPA_TDNN::loadweights(std::string pathname)
{
    assert(false && "Not completed");
};