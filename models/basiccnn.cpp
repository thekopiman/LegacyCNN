#include <iostream>
#include "basiccnn.h"
#include "../layers/basiccnnblock.h"
#include "../layers/helper.h"
#include "../layers/dense.h"

// This is broken for some reason and I'm not sure why
BasicCNNModel::BasicCNNModel()
{
    std::cout << "Initialising BasicCNNModel" << std::endl;
};
void BasicCNNModel::forward(float (&input)[2][16], float (&y)[6])
{
    this->Block0.getOutput(input, this->x0);
    this->Block1.getOutput(this->x0, this->x1);
    this->Block2.getOutput(this->x1, this->x2);
    this->Block3.getOutput(this->x2, this->x3);
    this->Block4.getOutput(this->x3, this->x4);
    Helper::Flatten(this->x4, this->flatten_x);
    this->fc.getOutput(this->flatten_x, y);
};