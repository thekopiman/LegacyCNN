/**
 * @file ecapa.cpp
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.1
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
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

    asp.forward(y0, y1);

    asp_BN.forward(y1, y1);

    MatrixFunctions::Flatten(y1, flatten_x);

    fc.forward(flatten_x, y);
};

ECAPA_TDNN::ECAPA_TDNN() : asp_BN(1)
{
    std::cout << "ECAPA TDNN initialised" << std::endl;
};

ECAPA_TDNN::~ECAPA_TDNN() {
};

void ECAPA_TDNN::loadweights(std::string pathname)
{
    std::ifstream infile(pathname, std::ios::binary);
    if (!infile)
    {
        std::cout << "Error opening file!" << std::endl;
        return;
    }

    // Initial Layer
    initiallayer.loadweights(infile);
    std::cout << "Initial Layer Loaded" << std::endl;

    // Seres
    seres_1.loadweights(infile);
    std::cout << "SERES 1 Loaded" << std::endl;
    seres_2.loadweights(infile);
    std::cout << "SERES 2 Loaded" << std::endl;
    seres_3.loadweights(infile);
    std::cout << "SERES 3 Loaded" << std::endl;

    // mfa
    mfa.loadweights(infile);
    std::cout << "mfa Loaded" << std::endl;

    // asp
    asp.loadweights(infile);
    std::cout << "asp Loaded" << std::endl;

    // ASP BN
    asp_BN.loadweights(infile);
    std::cout << "asp bn Loaded" << std::endl;

    // fc
    fc.loadweights(infile);
    std::cout << "fc Loaded" << std::endl;

    infile.close();
};