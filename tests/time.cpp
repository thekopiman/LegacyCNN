#include "time.h"

void Time::TestEcapa(int iter)
{
    assert(iter > 0);

    std::cout << "Testing Ecapa" << std::endl;

    // Initialise model
    auto start = std::chrono::high_resolution_clock::now();
    ECAPA_TDNN ecapamodel;
    float y[6];
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Model initialisation: " << duration.count() * 1000 << "ms" << std::endl;

    // Load weights
    start = std::chrono::high_resolution_clock::now();
    ecapamodel.loadweights("ECAPAweights/fullecapa.bin");
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Load weights: " << duration.count() * 1000 << "ms" << std::endl;

    // Metrics on Read Input will not be tested
    float input[2][64];
    Helper::readInputs("ECAPAweights/ecapainput_2x64.bin", input);

    // Forward feed
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; i++)
    {
        ecapamodel.forward(input, y);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = (end - start) / iter;
    std::cout << "Forward Feed (avg): " << duration.count() * 1000 << "ms" << std::endl;

    std::cout << std::endl;
}

void Time::TestBasic(int iter)
{

    std::cout << "Testing Basic Model" << std::endl;

    assert(iter > 0);

    // Initialise model
    auto start = std::chrono::high_resolution_clock::now();
    BasicCNNModel basicmodel;
    float y[6];
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Model initialisation: " << duration.count() * 1000 << "ms" << std::endl;

    // Load weights
    start = std::chrono::high_resolution_clock::now();
    basicmodel.loadweights("BasicModelWeights/fullbasicmodel.bin");
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Load weights: " << duration.count() * 1000 << "ms" << std::endl;

    // Metrics on Read Input will not be tested
    float input[2][16];
    Helper::readInputs("BasicModelWeights/basicinput_2x16.bin", input);

    // Forward feed
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; i++)
    {
        basicmodel.forward(input, y);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = (end - start) / iter;
    std::cout << "Forward Feed (avg): " << duration.count() * 1000 << "ms" << std::endl;

    std::cout << std::endl;
}
void Time::TestEcapaClassifier(int iter, int classNo, float lengths)
{
    std::cout << "Testing Ecapa Classifier Model with lengths" << std::endl;

    assert(iter > 0);
    assert(lengths > 0);
    assert(classNo == 0 || classNo == 1 || classNo == 2);

    // Initialise model
    auto start = std::chrono::high_resolution_clock::now();
    ECAPA_TDNN_classifier model(classNo);
    float y_2[6];     // For classno = 2 which is Euclidean
    float y_01[6][6]; // For classno = 0 or 1 which are Cosine and Cdist respectively
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Model initialisation: " << duration.count() * 1000 << "ms" << std::endl;

    // Load weights
    start = std::chrono::high_resolution_clock::now();
    model.loadweights("ECAPAweights/fullecapa_classifier.bin");
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Load weights: " << duration.count() * 1000 << "ms" << std::endl;

    // Metrics on Read Input will not be tested
    float input[2][64];
    Helper::readInputs("BasicModelWeights/ecapainput_2x64.bin", input);

    // Forward feed
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; i++)
    {
        if (classNo == 0 || classNo == 1)
        {
            model.forward(input, lengths, y_01);
        }
        else if (classNo == 2)
        {
            model.forward(input, lengths, y_2);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    duration = (end - start) / iter;
    std::cout << "Forward Feed (avg): " << duration.count() * 1000 << "ms" << std::endl;

    std::cout << std::endl;
}