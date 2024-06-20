#include "batchnorm1d.h"

template <int channel, int width, typename T>
BatchNorm1d<channel, width, T>::BatchNorm1d()
{
    for (int i = 0; i < channel; i++)
    {
        this->gamma[i] = 1;
        this->beta[i] = 0;
        this->mean[i] = 0;
        this->variance[i] = 0;
    }
}

template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::setGamma(T (&new_gamma)[channel])
{
    for (int i = 0; i < channel; i++)
    {
        this->gamma[i] = new_gamma[i];
    }
};

template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::setBeta(T (&new_beta)[channel])
{
    for (int i = 0; i < channel; i++)
    {
        this->beta[i] = new_beta[i];
    }
};

template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::setGamma(std::string filename)
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
    infile.read(reinterpret_cast<char *>(this->gamma), total_size * sizeof(T));
    infile.close();
};

template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::setBeta(std::string filename)
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
    infile.read(reinterpret_cast<char *>(this->beta), total_size * sizeof(T));
    infile.close();
};

template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::setGamma(std::ifstream &infile)
{
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
    infile.read(reinterpret_cast<char *>(this->gamma), total_size * sizeof(T));
};

template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::setBeta(std::ifstream &infile)
{
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
    infile.read(reinterpret_cast<char *>(this->beta), total_size * sizeof(T));
};

template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::setEps(T var)
{
    this->eps = var;
};

template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::forward(T (&input)[channel][width], T (&output)[channel][width])
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

template <int channel, int width, typename T>
BatchNorm1d<channel, width, T>::~BatchNorm1d(){};
