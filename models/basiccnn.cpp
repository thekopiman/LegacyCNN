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
    std::cout << "Model initialised" << std::endl;
};
BasicCNNModel::~BasicCNNModel(){};

void BasicCNNModel::loadweights()
{
    Block0.setBias_layer0("BasicModelWeights/layer0_conv_bias.bin", false);
    Block0.setWeights_layer0("BasicModelWeights/layer0_conv_weights.bin", false);
    Block0.setGamma_layer1("BasicModelWeights/layer0_bn_weights.bin");
    Block0.setBeta_layer1("BasicModelWeights/layer0_bn_bias.bin");

    Block1.setBias_layer0("BasicModelWeights/layer1_conv_bias.bin", false);
    Block1.setWeights_layer0("BasicModelWeights/layer1_conv_weights.bin", false);
    Block1.setGamma_layer1("BasicModelWeights/layer1_bn_weights.bin");
    Block1.setBeta_layer1("BasicModelWeights/layer1_bn_bias.bin");

    Block2.setBias_layer0("BasicModelWeights/layer2_conv_bias.bin", false);
    Block2.setWeights_layer0("BasicModelWeights/layer2_conv_weights.bin", false);
    Block2.setGamma_layer1("BasicModelWeights/layer2_bn_weights.bin");
    Block2.setBeta_layer1("BasicModelWeights/layer2_bn_bias.bin");

    Block3.setBias_layer0("BasicModelWeights/layer3_conv_bias.bin", false);
    Block3.setWeights_layer0("BasicModelWeights/layer3_conv_weights.bin", false);
    Block3.setGamma_layer1("BasicModelWeights/layer3_bn_weights.bin");
    Block3.setBeta_layer1("BasicModelWeights/layer3_bn_bias.bin");

    Block4.setBias_layer0("BasicModelWeights/layer4_conv_bias.bin", false);
    Block4.setWeights_layer0("BasicModelWeights/layer4_conv_weights.bin", false);
    Block4.setGamma_layer1("BasicModelWeights/layer4_bn_weights.bin");
    Block4.setBeta_layer1("BasicModelWeights/layer4_bn_bias.bin");

    fc.setBias("BasicModelWeights/final_fc_bias.bin");
    fc.setWeights("BasicModelWeights/final_fc_weights.bin", false);
}
