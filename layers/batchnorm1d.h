#ifndef batchnorm1d_h
#define batchnorm1d_h

#include <cmath>
#include <fstream>
#include <assert.h>
#include <string>
#include <iostream>
#include <vector>

// https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
template <int channel, int width>
class BatchNorm1d
{
public:
    BatchNorm1d()
    {
        // Initialise gamma as 1
        for (int i = 0; i < channel; i++)
        {
            this->gamma[i] = 1.0f;
        }

        // Initialise beta as 0
        for (int i = 0; i < channel; i++)
        {
            this->beta[i] = 0.0f;
        }

        // Initialise mean as 0
        for (int i = 0; i < channel; i++)
        {
            this->mean[i] = 0.0f;
        }

        // Initialise variance as 0
        for (int i = 0; i < channel; i++)
        {
            this->variance[i] = 0.0f;
        }
    };
    void setGamma(float (&new_gamma)[channel])
    {
        // Initialise gamma as 1
        for (int i = 0; i < channel; i++)
        {
            this->gamma[i] = new_gamma[i];
        }
    };
    void setBeta(float (&new_beta)[channel])
    {
        // Initialise beta as 1
        for (int i = 0; i < channel; i++)
        {
            this->beta[i] = new_beta[i];
        }
    };

    // Overloading
    void setGamma(std::string filename)
    {
        std::ifstream infile(filename, std::ios::binary);
        if (!infile)
        {
            std::cout << "Error opening file!" << std::endl;
            return;
        }
        // Read dimensions
        int dim1;
        infile.read(reinterpret_cast<char *>(&dim1), sizeof(int));

        // Sanity Check
        assert(dim1 == channel);

        // Calculate total size
        int total_size = dim1;

        // Read flattened array
        std::vector<float> flat_array(total_size);
        infile.read(reinterpret_cast<char *>(flat_array.data()), total_size * sizeof(float));
        infile.close();

        for (int i = 0; i < dim1; ++i)
        {
            this->gamma[i] = flat_array[i];
        }
    };

    // Overloading
    void setBeta(std::string filename)
    {
        std::ifstream infile(filename, std::ios::binary);
        if (!infile)
        {
            std::cout << "Error opening file!" << std::endl;
            return;
        }
        // Read dimensions
        int dim1;
        infile.read(reinterpret_cast<char *>(&dim1), sizeof(int));

        // Sanity Check
        assert(dim1 == channel);

        // Calculate total size
        int total_size = dim1;

        // Read flattened array
        std::vector<float> flat_array(total_size);
        infile.read(reinterpret_cast<char *>(flat_array.data()), total_size * sizeof(float));
        infile.close();

        for (int i = 0; i < dim1; ++i)
        {
            this->beta[i] = flat_array[i];
        }
    };

    void setEps(float var)
    {
        this->eps = var;
    };
    void getOutput(float (&input)[channel][width], float (&output)[channel][width])
    {
        // Calculate mean E(X) first
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < width; i++)
            {
                this->mean[c] += input[c][i];
            }
            this->mean[c] /= width;
        }

        // Calculate Population Variance Var(X) = 1/n sum(x - E(x))^2
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < width; i++)
            {
                this->variance[c] += (input[c][i] - this->mean[c]) * (input[c][i] - this->mean[c]);
            }
            this->variance[c] /= width;
        }

        // Normalize and scale + shift
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < width; i++)
            {
                output[c][i] = this->gamma[c] * (input[c][i] - this->mean[c]) / std::sqrt(this->variance[c] + this->eps) + this->beta[c];
            }
        }
    };

private:
    float gamma[channel];
    float beta[channel];
    float mean[channel];
    float variance[channel];
    float eps = 1e-5;
};

#endif