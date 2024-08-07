/**
 * @file batchnorm1d.cpp
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.1
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "batchnorm1d.h"
/**
 * @brief Construct a new Batch Norm 1d<channel, width,  T>:: Batch Norm 1d object
 *
 * It will default to train mode
 *
 * @tparam channel
 * @tparam width
 * @tparam T
 */
template <int channel, int width, typename T>
BatchNorm1d<channel, width, T>::BatchNorm1d()
{
    init();
}

/**
 * @brief Construct a new Batch Norm 1d<channel, width,  T>:: Batch Norm 1d object
 *
 * @tparam channel
 * @tparam width
 * @tparam T
 * @param mode: defaults to 0 (train mode). 1 is eval mode
 */
template <int channel, int width, typename T>
BatchNorm1d<channel, width, T>::BatchNorm1d(int mode)
{
    if (mode == 1)
    {
        this->eval_mode = true;
    }
    init();
}

/**
 * @brief The `init()` function in the `BatchNorm1d` class template is initializing the parameters for each
channel. It sets the initial values for `gamma`, `beta`, `mean`, `variance`, `running_mean`, and
`running_variance` arrays for each channel.
 *
 * @tparam channel
 * @tparam width
 * @tparam T
 */
template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::init()
{
    for (int i = 0; i < channel; i++)
    {
        this->gamma[i] = 1;
        this->beta[i] = 0;
        this->mean[i] = 0;
        this->variance[i] = 0;
        this->running_mean[i] = 0;
        this->running_variance[i] = 0;
    }
}

/**
 * @brief The `init_temp()` function in the `BatchNorm1d` class template is initializing the parameters for each
channel. It sets the initial values for `mean` and `variance` arrays for each channel.
*
* @tparam channel
* @tparam width
* @tparam T
*/
template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::init_temp()
{
    for (int i = 0; i < channel; i++)
    {
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
void BatchNorm1d<channel, width, T>::setMean(T (&new_mean)[channel])
{
    for (int i = 0; i < channel; i++)
    {
        this->running_mean[i] = new_mean[i];
    }
};

template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::setVar(T (&new_var)[channel])
{
    for (int i = 0; i < channel; i++)
    {
        this->running_variance[i] = new_var[i];
    }
};

template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::setGamma(std::string pathname)
{
    std::ifstream infile(pathname, std::ios::binary);
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
void BatchNorm1d<channel, width, T>::setBeta(std::string pathname)
{
    std::ifstream infile(pathname, std::ios::binary);
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
void BatchNorm1d<channel, width, T>::setMean(std::string pathname)
{
    std::ifstream infile(pathname, std::ios::binary);
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
    infile.read(reinterpret_cast<char *>(this->running_mean), total_size * sizeof(T));
    infile.close();
};

template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::setVar(std::string pathname)
{
    std::ifstream infile(pathname, std::ios::binary);
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
    infile.read(reinterpret_cast<char *>(this->running_variance), total_size * sizeof(T));
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
void BatchNorm1d<channel, width, T>::setMean(std::ifstream &infile)
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
    infile.read(reinterpret_cast<char *>(this->running_mean), total_size * sizeof(T));
};

template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::setVar(std::ifstream &infile)
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
    infile.read(reinterpret_cast<char *>(this->running_variance), total_size * sizeof(T));
};

/**
 * @brief Load the weights via infile
 *
 * @tparam channel
 * @tparam width
 * @tparam T
 * @param infile
 */
template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::loadweights(std::ifstream &infile)
{
    setGamma(infile);
    setBeta(infile);
    setMean(infile);
    setVar(infile);
};

/**
 * @brief Load the weights via pathname
 *
 * @tparam channel
 * @tparam width
 * @tparam T
 * @param pathname
 */
template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::loadweights(std::string pathname)
{
    std::ifstream infile(pathname, std::ios::binary);
    if (!infile)
    {
        std::cout << "Error opening file!" << std::endl;
        return;
    }
    setGamma(infile);
    setBeta(infile);
    setMean(infile);
    setVar(infile);
};

template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::setEps(T var)
{
    this->eps = var;
};

/**
 * @brief Forward feed during train mode
 *
 * @tparam channel
 * @tparam width
 * @tparam T
 * @param input
 * @param output
 */
template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::forward_train(T (&input)[channel][width], T (&output)[channel][width])
{
    init_temp();

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

/**
 * @brief Forward feed during eval mode
 *
 * @tparam channel
 * @tparam width
 * @tparam T
 * @param input
 * @param output
 */
template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::forward_eval(T (&input)[channel][width], T (&output)[channel][width])
{
    // Normalize and scale + shift
    for (int c = 0; c < channel; c++)
    {
        for (int i = 0; i < width; i++)
        {
            output[c][i] = this->gamma[c] * (input[c][i] - this->running_mean[c]) / std::sqrt(this->running_variance[c] + this->eps) + this->beta[c];
        }
    }
};

/**
 * @brief Performs forward feed in accordance to the mode being set during construction.
 *
 * @tparam channel
 * @tparam width
 * @tparam T
 */
template <int channel, int width, typename T>
void BatchNorm1d<channel, width, T>::forward(T (&input)[channel][width], T (&output)[channel][width])
{
    if (this->eval_mode)
    {
        forward_eval(input, output);
    }
    else
    {
        forward_train(input, output);
    }
};

template <int channel, int width, typename T>
BatchNorm1d<channel, width, T>::~BatchNorm1d(){};
