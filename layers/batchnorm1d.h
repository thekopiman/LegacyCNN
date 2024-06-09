#ifndef batchnorm1d_h
#define batchnorm1d_h

#include <cmath>
#include <fstream>
#include <assert.h>
#include <string>
#include <iostream>

// https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
template <int channel, int width, typename T>
class BatchNorm1d
{
public:
    BatchNorm1d()
    {
        // Initialise gamma as 1
        for (int i = 0; i < channel; i++)
        {
            this->gamma[i] = 1.0;
        }

        // Initialise beta as 0
        for (int i = 0; i < channel; i++)
        {
            this->beta[i] = 0.0;
        }

        // Initialise mean as 0
        for (int i = 0; i < channel; i++)
        {
            this->mean[i] = 0.0;
        }

        // Initialise variance as 0
        for (int i = 0; i < channel; i++)
        {
            this->variance[i] = 0.0;
        }
    };
    void setGamma(T (&new_gamma)[channel])
    {
        // Initialise gamma as 1
        for (int i = 0; i < channel; i++)
        {
            this->gamma[i] = new_gamma[i];
        }
    };
    void setBeta(T (&new_beta)[channel])
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
        infile.read(reinterpret_cast<char *>(this->temp_gamma), total_size * sizeof(float));
        infile.close();
        for (int i = 0; i < channel; i++)
        {
            this->gamma[i] = (T)this->temp_gamma[i];
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
        infile.read(reinterpret_cast<char *>(this->temp_beta), total_size * sizeof(float));
        infile.close();
        for (int i = 0; i < channel; i++)
        {
            this->beta[i] = (T)this->temp_beta[i];
        }
    };

    void setEps(T var)
    {
        this->eps = var;
    };
    void getOutput(T (&input)[channel][width], T (&output)[channel][width])
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
    T gamma[channel];
    T beta[channel];
    float temp_gamma[channel];
    float temp_beta[channel];
    T mean[channel];
    T variance[channel];
    T eps = 1e-5;
};

#endif