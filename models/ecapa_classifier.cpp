/**
 * @file ecapa_classifier.cpp
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.3
 * @date 2024-08-12
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "ecapa_classifier.h"

/**
 * @brief Forward for CDist/Cosine without lengths
 *
 * @param input
 * @param y
 */
void ECAPA_TDNN_classifier::forward(float (&input)[2][64], float (&y)[6][6])
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

    fc.forward(y1, y2);

    if (ClassifierNumber == 0)
    {
        CosineClassifier.forward(y2, y);
    }
    else if (ClassifierNumber == 1)
    {
        CDistClassifier.forward(y2, y);
    }
    else
    {
        assert(false && "Classifier Number must be 0, 1 or 2.");
    }

    ActivationFunctions::Softmax(y);
};

/**
 * @brief Forward for Euclidean without lengths
 *
 * @param input
 * @param y
 */
void ECAPA_TDNN_classifier::forward(float (&input)[2][64], float (&y)[6])
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

    fc.forward(y1, y2);

    if (ClassifierNumber == 2)
    {
        EuclideanClassifier.forward(y2, y);
    }
    else
    {
        assert(false && "Classifier Number must be 0, 1 or 2.");
    }

    ActivationFunctions::Softmax(y);
};

/**
 * @brief Forward for CDist/Cosine with lengths
 *
 * @param input
 * @param lengths
 * @param y
 */
void ECAPA_TDNN_classifier::forward(float (&input)[2][64], float &lengths, float (&y)[6][6])
{
    initiallayer.forward(input, x0);

    seres_1.forward(x0, lengths, x1);

    seres_2.forward(x1, lengths, x2);

    seres_3.forward(x2, lengths, x3);

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

    asp.forward(y0, lengths, y1);

    asp_BN.forward(y1, y1);

    fc.forward(y1, y2);

    if (ClassifierNumber == 0)
    {
        CosineClassifier.forward(y2, y);
    }
    else if (ClassifierNumber == 1)
    {
        CDistClassifier.forward(y2, y);
    }
    else
    {
        assert(false && "Classifier Number must be 0, 1 or 2.");
    }

    ActivationFunctions::Softmax(y);
};

/**
 * @brief Forward for EuclideanClassifier with Lengths
 *
 * @param input
 * @param lengths
 * @param y
 */
void ECAPA_TDNN_classifier::forward(float (&input)[2][64], float &lengths, float (&y)[6])
{
    initiallayer.forward(input, x0);

    seres_1.forward(x0, lengths, x1);

    seres_2.forward(x1, lengths, x2);

    seres_3.forward(x2, lengths, x3);

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

    asp.forward(y0, lengths, y1);

    asp_BN.forward(y1, y1);

    fc.forward(y1, y2);

    if (ClassifierNumber == 2)
    {
        EuclideanClassifier.forward(y2, y);
    }
    else
    {
        assert(false && "Classifier Number must be 0, 1 or 2.");
    }

    ActivationFunctions::Softmax(y);
};

ECAPA_TDNN_classifier::ECAPA_TDNN_classifier(int classno) : asp_BN(1), ClassifierNumber(classno)
{
    std::cout << "ECAPA TDNN Classifier initialised in Eval Mode" << std::endl;
};

ECAPA_TDNN_classifier::~ECAPA_TDNN_classifier() {
};

void ECAPA_TDNN_classifier::loadweights(std::string pathname)
{
    std::ifstream infile(pathname, std::ios::binary);
    if (!infile)
    {
        std::cout << "Error opening file!" << std::endl;
        return;
    }

    // Initial Layer
    initiallayer.loadweights(infile);
    // std::cout << "Initial Layer Loaded" << std::endl;

    // Seres
    seres_1.loadweights(infile);
    // std::cout << "SERES 1 Loaded" << std::endl;
    seres_2.loadweights(infile);
    // std::cout << "SERES 2 Loaded" << std::endl;
    seres_3.loadweights(infile);
    // std::cout << "SERES 3 Loaded" << std::endl;

    // mfa
    mfa.loadweights(infile);
    // std::cout << "mfa Loaded" << std::endl;

    // asp
    asp.loadweights(infile);
    // std::cout << "asp Loaded" << std::endl;

    // ASP BN
    asp_BN.loadweights(infile);
    // std::cout << "asp bn Loaded" << std::endl;

    // fc
    fc.loadweights(infile);
    // std::cout << "fc Loaded" << std::endl;

    // Classifier
    if (this->ClassifierNumber == 0)
    {
        CosineClassifier.loadweights(infile);
        // std::cout << "CosineClassifier Loaded" << std::endl;
    }
    else if (this->ClassifierNumber == 1)
    {
        CDistClassifier.loadweights(infile);
        // std::cout << "CDistClassifier Loaded" << std::endl;
    }
    else if (this->ClassifierNumber == 2)
    {
        EuclideanClassifier.loadweights(infile);
        // std::cout << "EuclideanClassifier Loaded" << std::endl;
    }
    else
    {
        assert(false && "Classifier Number must be 0, 1 or 2.");
    }

    infile.close();

    std::cout << "ECAPA TDNN weights loaded successfully." << std::endl;
};