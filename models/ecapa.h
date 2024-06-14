#ifndef ecapatdnnmodel_h
#define ecapatdnnmodel_h

#include <iostream>

#include "../layers/conv1d.h"
#include "../layers/dense.h"
#include "../layers/tdnnblock.h"
#include "../layers/res2netblock.h"
#include "../layers/seblock.h"
#include "../layers/seres2netblock.h"
#include "../layers/batchnorm1d.h"
#include "../utils/helper.h"
#include "../utils/activationfunctions.h"
#include "../utils/matrixfunctions.h"

class ECAPA_TDNN
{
public:
    ECAPA_TDNN();
    ~ECAPA_TDNN();
    void forward(float (&input)[2][64], float (&y)[6]);
    void loadweights();

private:
    TDNNBlock<5, 1, 2, 8, 1, 64, 64, 4, float> initiallayer;
    SERes2NetBlock<3, 8, 8, 2, 64, 64, 4, 8, 128, float> seres_1;
    SERes2NetBlock<3, 8, 8, 3, 64, 64, 6, 8, 128, float> seres_2;
    SERes2NetBlock<3, 8, 8, 4, 64, 64, 8, 8, 128, float> seres_3;
    TDNNBlock<1, 1, 3 * 8, 16, 1, 64, 64, 0, float> mfa;
    // ASP is not done
    BatchNorm1d<16 * 2, 64, float> ASP_BN;
    // Flatten()
    float flatten_x[16 * 2];
    Dense<16 * 2, 6, float> fc;
};

#endif