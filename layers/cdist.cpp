/**
 * @file cdist.cpp
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.3
 * @date 2024-08-12
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "cdist.h"

/**
 * @brief Construct a new cdist<channels, input width, out width, t>::cdist object
 *
 * @tparam channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 */
template <int channels, int input_width, int out_width, typename T>
CDist<channels, input_width, out_width, T>::CDist(){};

/**
 * @brief Destroy the cdist<channels, input width, out width, t>::cdist object
 *
 * @tparam channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 */
template <int channels, int input_width, int out_width, typename T>
CDist<channels, input_width, out_width, T>::~CDist(){};

/**
 * @brief Performs CDist with variable p (integer) where p > 0
 *
 * @tparam channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 * @param input
 * @param output
 * @param p
 */
template <int channels, int input_width, int out_width, typename T>
void CDist<channels, input_width, out_width, T>::forward(T (&input)[channels][input_width], T (&output)[channels][out_width], int p)
{
    MatrixFunctions::CDist(input, this->weights, output, p);
};

/**
 * @brief Performs CDist where p = 2
 *
 * @tparam channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 * @param input
 * @param output
 */
template <int channels, int input_width, int out_width, typename T>
void CDist<channels, input_width, out_width, T>::forward(T (&input)[channels][input_width], T (&output)[channels][out_width])
{
    MatrixFunctions::CDist(input, this->weights, output, 2);
};

/**
 * @brief Loadweights via infile
 *
 * @tparam channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 * @param infile
 */
template <int channels, int input_width, int out_width, typename T>
void CDist<channels, input_width, out_width, T>::loadweights(std::ifstream &infile)
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
    assert(dim1 == out_width);
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
 * @tparam out_width
 * @tparam T
 * @param pathname
 */
template <int channels, int input_width, int out_width, typename T>
void CDist<channels, input_width, out_width, T>::loadweights(std::string pathname)
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