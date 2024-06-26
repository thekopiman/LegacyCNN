/**
 * @file basiccnn.h
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.1
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef basiccnnmodel_h
#define basiccnnmodel_h

#include <iostream>

#include "../layers/conv1d.h"
#include "../layers/dense.h"
#include "../utils/helper.h"
#include "../utils/activationfunctions.h"
#include "../utils/matrixfunctions.h"
#include "../layers/basicblock.h"
#include "../layers/dense.h"

class BasicCNNModel
{
public:
    BasicCNNModel();
    ~BasicCNNModel();
    void forward(float (&input)[2][16], float (&y)[6]);
    void loadweights(std::string pathname);

private:
    BasicBlock<3, 1, 2, 4, 0, 1, 16, 14, float> Block0;
    float x0[4][14];
    BasicBlock<3, 1, 4, 4, 0, 1, 14, 12, float> Block1;
    float x1[4][12];
    BasicBlock<3, 1, 4, 4, 0, 1, 12, 10, float> Block2;
    float x2[4][10];
    BasicBlock<3, 1, 4, 4, 0, 1, 10, 8, float> Block3;
    float x3[4][8];
    BasicBlock<3, 1, 4, 4, 0, 1, 8, 6, float> Block4;
    float x4[4][6];
    // Flatten()
    float flatten_x[24];
    Dense<24, 6, float> fc;
};

#include "basiccnn.cpp"

#endif
