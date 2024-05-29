#include <iostream>
#include <vector>
#include <cmath>

// Constants
const int CHANNELS = 2;
const int FEATURE_SIZE = 64;
const float epsilon = 1e-5;

// Function to perform batch normalization
void batchNormalization(float input[CHANNELS][FEATURE_SIZE], float output[CHANNELS][FEATURE_SIZE], float gamma[CHANNELS][FEATURE_SIZE], float beta[CHANNELS][FEATURE_SIZE])
{
    float mean[CHANNELS] = {0};
    float variance[CHANNELS] = {0};

    // Calculate mean for each channel
    for (int c = 0; c < CHANNELS; ++c)
    {
        for (int j = 0; j < FEATURE_SIZE; ++j)
        {
            mean[c] += input[c][j];
        }
        mean[c] /= FEATURE_SIZE;
    }

    // Calculate variance for each channel
    for (int c = 0; c < CHANNELS; ++c)
    {
        for (int j = 0; j < FEATURE_SIZE; ++j)
        {
            variance[c] += (input[c][j] - mean[c]) * (input[c][j] - mean[c]);
        }
        variance[c] /= FEATURE_SIZE;
    }

    // Normalize and scale + shift
    for (int c = 0; c < CHANNELS; ++c)
    {
        for (int j = 0; j < FEATURE_SIZE; ++j)
        {
            output[c][j] = gamma[c][j] * (input[c][j] - mean[c]) / std::sqrt(variance[c] + epsilon) + beta[c][j];
        }
    }
}

int main()
{
    // Example input: 1x2x64 matrix (flattened to 2x64 for simplicity)
    float input[CHANNELS][FEATURE_SIZE] = {
        {/* ... 64 values for channel 1 ... */},
        {/* ... 64 values for channel 2 ... */}};

    // Example gamma and beta (scale and shift parameters)
    float gamma[CHANNELS][FEATURE_SIZE];
    float beta[CHANNELS][FEATURE_SIZE];
    for (int c = 0; c < CHANNELS; ++c)
    {
        for (int j = 0; j < FEATURE_SIZE; ++j)
        {
            gamma[c][j] = 1.0; // Typically initialized to 1
            beta[c][j] = 0.0;  // Typically initialized to 0
        }
    }

    // Output array
    float output[CHANNELS][FEATURE_SIZE];

    // Perform batch normalization
    batchNormalization(input, output, gamma, beta);

    // Print output for verification
    std::cout << "Normalized Output:\n";
    for (int c = 0; c < CHANNELS; ++c)
    {
        for (int j = 0; j < FEATURE_SIZE; ++j)
        {
            std::cout << output[c][j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
