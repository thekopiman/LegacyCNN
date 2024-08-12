/**
 * @file cosinesimilarity.cpp
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.3
 * @date 2024-08-12
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "cosinesimilarity.h"

/**
 * @brief Construct a new CosineSimilarity<channels, input_width, out_width, T>::CosineSimilarity object
 *
 * @tparam channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 */
template <int channels, int input_width, int out_width, typename T>
CosineSimilarity<channels, input_width, out_width, T>::CosineSimilarity(){};

/**
 * @brief Destroy the CosineSimilarity<channels, input_width, out_width, T>::CosineSimilarity object
 *
 * @tparam channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 */
template <int channels, int input_width, int out_width, typename T>
CosineSimilarity<channels, input_width, out_width, T>::~CosineSimilarity(){};

/**
 * @brief Performs forwardfeed. Bias = None
 *
 * @tparam channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 * @param input
 * @param output
 */
template <int channels, int input_width, int out_width, typename T>
void CosineSimilarity<channels, input_width, out_width, T>::forward(T (&input)[channels][input_width], T (&output)[channels][out_width])
{
    // Make a duplicate since L2Norm is inplace
    MatrixFunctions::Copy(input, this->input_copy);

    MatrixFunctions::L2Normalisation(input_copy);
    // L2Norm + Transpose has been done during weights loading

    MatrixFunctions::DotProduct(input_copy, this->weights_T, output);
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
void CosineSimilarity<channels, input_width, out_width, T>::loadweights(std::string pathname)
{
    std::ifstream infile(pathname, std::ios::binary);
    if (!infile)
    {
        std::cout << "Error opening file!" << std::endl;
        return;
    }

    loadweights(infile);

    infile.close();
}

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
void CosineSimilarity<channels, input_width, out_width, T>::loadweights(std::ifstream &infile)
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

    MatrixFunctions::L2Normalisation(this->weights);
    MatrixFunctions::Transpose(this->weights, this->weights_T);
}