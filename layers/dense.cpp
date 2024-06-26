/**
 * @file dense.cpp
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.1
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "dense.h"

/**
 * @brief Construct a new Dense<input_dim, output_dim, T>::Dense object
 *
 * @tparam input_dim
 * @tparam output_dim
 * @tparam T
 */
template <int input_dim, int output_dim, typename T>
Dense<input_dim, output_dim, T>::Dense()
{
    // Initialize the weights with 0
    for (int i = 0; i < output_dim; i++)
    {
        for (int j = 0; j < input_dim; j++)
        {
            this->weights[i][j] = 0.0;
        }
    }

    // Initialize the bias with 0
    for (int i = 0; i < output_dim; i++)
    {
        this->bias[i] = 0.0;
    }

    // std::cout << "Dense initialised" << std::endl;
}

template <int input_dim, int output_dim, typename T>
void Dense<input_dim, output_dim, T>::setWeights(T (&new_weights)[input_dim][output_dim])
{
    // New weights are being set
    for (int i = 0; i < input_dim; i++)
    {
        for (int j = 0; j < output_dim; j++)
        {
            // std::cout << i << j << " " << new_weights[i][j] << std::endl;
            this->weights[i][j] = new_weights[i][j];
        }
    }
};

template <int input_dim, int output_dim, typename T>
void Dense<input_dim, output_dim, T>::setBias(T (&new_bias)[output_dim])
{
    // New bias are being set
    for (int i = 0; i < output_dim; i++)
    {
        this->bias[i] = new_bias[i];
    }
};

template <int input_dim, int output_dim, typename T>
void Dense<input_dim, output_dim, T>::setBias(std::string pathname)
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
    assert(dim1 == output_dim);

    // Calculate total size
    int total_size = dim1;

    // Read flattened array
    infile.read(reinterpret_cast<char *>(this->bias), total_size * sizeof(T));
    infile.close();
};

template <int input_dim, int output_dim, typename T>
void Dense<input_dim, output_dim, T>::setWeights(std::string pathname, bool displayWeights)
{
    std::ifstream infile(pathname, std::ios::binary);
    if (!infile)
    {
        std::cout << "Error opening file!" << std::endl;
        return;
    }
    // Read dimensions
    int dim1, dim2;
    infile.read(reinterpret_cast<char *>(&dim1), sizeof(int));
    infile.read(reinterpret_cast<char *>(&dim2), sizeof(int));

    // Sanity Check
    assert(dim1 == output_dim);
    assert(dim2 == input_dim);

    // Calculate total size
    int total_size = dim1 * dim2;

    // Read flattened array
    infile.read(reinterpret_cast<char *>(this->flat_weights), total_size * sizeof(T));
    infile.close();

    for (int i = 0; i < dim1; ++i)
    {
        for (int j = 0; j < dim2; ++j)
        {
            this->weights[i][j] = (T)this->flat_weights[i * dim2 + j];
        }
    }

    if (displayWeights)
    {
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                std::cout << this->weights[i][j] << " ";
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
    }
};

template <int input_dim, int output_dim, typename T>
void Dense<input_dim, output_dim, T>::setWeights(std::ifstream &infile, bool displayWeights)
{
    // Read dimensions
    int dim1, dim2;
    infile.read(reinterpret_cast<char *>(&dim1), sizeof(int));
    infile.read(reinterpret_cast<char *>(&dim2), sizeof(int));

    // Sanity Check
    assert(dim1 == output_dim);
    assert(dim2 == input_dim);

    // Calculate total size
    int total_size = dim1 * dim2;

    // Read flattened array
    infile.read(reinterpret_cast<char *>(this->flat_weights), total_size * sizeof(T));

    for (int i = 0; i < dim1; ++i)
    {
        for (int j = 0; j < dim2; ++j)
        {
            this->weights[i][j] = (T)this->flat_weights[i * dim2 + j];
        }
    }

    if (displayWeights)
    {
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                std::cout << this->weights[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

/**
 * @brief Performs forward feed
 *
 * @tparam input_dim
 * @tparam output_dim
 * @tparam T
 * @param input
 * @param output
 */
template <int input_dim, int output_dim, typename T>
void Dense<input_dim, output_dim, T>::forward(T (&input)[input_dim], T (&output)[output_dim])
{
    for (int i = 0; i < output_dim; i++)
    {
        output[i] = this->bias[i];
        for (int j = 0; j < input_dim; j++)
        {
            output[i] += this->weights[i][j] * input[j];
        }
    }
}

template <int input_dim, int output_dim, typename T>
Dense<input_dim, output_dim, T>::~Dense() {}

template <int input_dim, int output_dim, typename T>
void Dense<input_dim, output_dim, T>::setBias(std::ifstream &infile)
{
    // Read dimensions
    int dim1;
    infile.read(reinterpret_cast<char *>(&dim1), sizeof(int));

    // Sanity Check
    assert(dim1 == output_dim);

    // Calculate total size
    int total_size = dim1;

    // Read flattened array
    infile.read(reinterpret_cast<char *>(this->bias), total_size * sizeof(T));
};

/**
 * @brief Load weights via infile
 *
 * @tparam input_dim
 * @tparam output_dim
 * @tparam T
 * @param infile
 */
template <int input_dim, int output_dim, typename T>
void Dense<input_dim, output_dim, T>::loadweights(std::ifstream &infile)
{
    setWeights(infile, false);
    setBias(infile);
};

/**
 * @brief Load weights via pathname
 *
 * @tparam input_dim
 * @tparam output_dim
 * @tparam T
 * @param pathname
 */
template <int input_dim, int output_dim, typename T>
void Dense<input_dim, output_dim, T>::loadweights(std::string pathname)
{
    std::ifstream infile(pathname, std::ios::binary);
    if (!infile)
    {
        std::cout << "Error opening file!" << std::endl;
        return;
    }
    setWeights(infile, false);
    setBias(infile);

    infile.close();
};