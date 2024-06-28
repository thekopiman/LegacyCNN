/**
 * @file basiccnn.cpp
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.1
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "basiccnn.h"

void BasicCNNModel::forward(float (&input)[2][16], float (&y)[6])
{
    Block0.forward(input, x0);

    Block1.forward(x0, x1);

    Block2.forward(x1, x2);

    Block3.forward(x2, x3);

    Block4.forward(x3, x4);

    MatrixFunctions::Flatten(x4, flatten_x);

    fc.forward(flatten_x, y);

    std::cout << std::endl;

    ActivationFunctions::Softmax(y);
};

BasicCNNModel::BasicCNNModel()
{
    std::cout << "BasicModel initialised" << std::endl;
};
BasicCNNModel::~BasicCNNModel() {};

void BasicCNNModel::loadweights(std::string pathname)
{
    std::ifstream infile(pathname, std::ios::binary);
    if (!infile)
    {
        std::cout << "Error opening file!" << std::endl;
        return;
    }

    Block0.loadweights(infile);
    Block1.loadweights(infile);
    Block2.loadweights(infile);
    Block3.loadweights(infile);
    Block4.loadweights(infile);
    fc.loadweights(infile);
}
