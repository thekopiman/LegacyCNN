/**
 * @file euclidean.cpp
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.3
 * @date 2024-08-13
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "euclidean.h"
/**
 * @brief Construct a new Euclidean<channels, input_width, T>::Euclidean object
 *
 * @tparam channels
 * @tparam input_width
 * @tparam T
 */
template <int channels, int input_width, typename T>
Euclidean<channels, input_width, T>::Euclidean(){};

/**
 * @brief Destroy the Euclidean<channels, input_width, T>::Euclidean object
 *
 * @tparam channels
 * @tparam input_width
 * @tparam T
 */
template <int channels, int input_width, typename T>
Euclidean<channels, input_width, T>::~Euclidean(){};

/**
 * @brief Loadweights via infile
 *
 * @tparam channels
 * @tparam input_width
 * @tparam T
 * @param infile
 */
template <int channels, int input_width, typename T>
void Euclidean<channels, input_width, T>::loadweights(std::ifstream &infile)
{
    if (!infile)
    {
        std::cout << "Error opening file!" << std::endl;
        return;
    }

    int dim1, dim2;
    infile.read(reinterpret_cast<char *>(&dim1), sizeof(int));
    infile.read(reinterpret_cast<char *>(&dim2), sizeof(int));

    // Sanity Check
    assert(dim1 == channels);
    assert(dim2 == input_width);

    // Calculate total size
    int total_size = dim1 * dim2;

    // Read flattened array
    infile.read(reinterpret_cast<char *>(this->flat_matrix), total_size * sizeof(T));

    for (int i = 0; i < dim1; ++i)
    {
        for (int j = 0; j < dim2; ++j)
        {
            this->weights[i][j] = (T)this->flat_matrix[i * dim2 + j];
        }
    }
};

/**
 * @brief Loadweights via pathname
 *
 * @tparam channels
 * @tparam input_width
 * @tparam T
 * @param pathname
 */
template <int channels, int input_width, typename T>
void Euclidean<channels, input_width, T>::loadweights(std::string pathname)
{
    std::ifstream infile(pathname, std::ios::binary);
    if (!infile)
    {
        std::cout << "Error opening file!" << std::endl;
        return;
    }

    loadweights(infile);

    infile.close();
};

/**
 * @brief Performs forward feed.
 *
 * @tparam channels
 * @tparam input_width
 * @tparam T
 * @param input
 * @param output
 */
template <int channels, int input_width, typename T>
void Euclidean<channels, input_width, T>::forward(T (&input)[channels][input_width], T (&output)[channels])
{
    for (int i = 0; i < channels; i++)
    {
        output[i] = 0; // Reset

        // Sum (input - weight)
        for (int j = 0; j < input_width; j++)
        {
            output[i] = (input[i][j] - this->weights[i][j]) * (input[i][j] - this->weights[i][j]);
        }

        // Sqrt
        output[i] = std::sqrt(output[i]);
    }
}