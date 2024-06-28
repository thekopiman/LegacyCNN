/**
 * @file ecapa.h
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.1
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef ecapatdnnmodel_h
#define ecapatdnnmodel_h

#include <iostream>
#include <string>
#include <assert.h>
#include <fstream>

#include "../layers/conv1d.h"
#include "../layers/dense.h"
#include "../layers/tdnnblock.h"
#include "../layers/res2netblock.h"
#include "../layers/seblock.h"
#include "../layers/seres2netblock.h"
#include "../layers/batchnorm1d.h"
#include "../layers/asp.h"
#include "../utils/helper.h"
#include "../utils/activationfunctions.h"
#include "../utils/matrixfunctions.h"

class ECAPA_TDNN
{
public:
    ECAPA_TDNN();
    ~ECAPA_TDNN();
    void forward(float (&input)[2][64], float (&y)[6]);
    void loadweights(std::string pathname);

private:
    TDNNBlock<5, 1, 2, 8, 1, 64, 64, 4, float> initiallayer;
    float x0[8][64];
    SERes2NetBlock<3, 8, 8, 2, 64, 64, 4, 8, 128, float> seres_1;
    float x1[8][64];
    SERes2NetBlock<3, 8, 8, 3, 64, 64, 6, 8, 128, float> seres_2;
    float x2[8][64];
    SERes2NetBlock<3, 8, 8, 4, 64, 64, 8, 8, 128, float> seres_3;
    float x3[8][64];
    float x_cat[8 * 3][64];
    TDNNBlock<1, 1, 3 * 8, 16, 1, 64, 64, 0, float> mfa;
    float y0[16][64];
    ASP<16, 128, 64, 2, float> asp;
    float y1[32][1];
    BatchNorm1d<16 * 2, 1, float> asp_BN;
    // Flatten()
    float flatten_x[16 * 2];
    Dense<16 * 2, 6, float> fc;
};

#include "ecapa.cpp"

#endif