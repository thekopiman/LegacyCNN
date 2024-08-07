/**
 * @file helper.cpp
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief Extra functions
 * @version 0.1
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "helper.h"

/**
 * @brief Read the .bin file and returns it to input
 *
 * @tparam rows
 * @tparam cols
 * @tparam T
 * @param pathname
 * @param inputs
 */
template <size_t rows, size_t cols, typename T>
void Helper::readInputs(std::string pathname, T (&inputs)[rows][cols])
{
    float flat_matrix[rows * cols];

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
    assert(dim1 == rows);
    assert(dim2 == cols);

    // Calculate total size
    int total_size = dim1 * dim2;

    // Read flattened array
    infile.read(reinterpret_cast<char *>(flat_matrix), total_size * sizeof(T));
    infile.close();

    for (int i = 0; i < dim1; ++i)
    {
        for (int j = 0; j < dim2; ++j)
        {
            inputs[i][j] = (T)flat_matrix[i * dim2 + j];
        }
    }
}

/**
 * @brief Read the .bin file and returns it to input
 *
 * @tparam rows
 * @tparam cols
 * @tparam T
 * @param pathname
 * @param inputs
 */
template <size_t batch, size_t rows, size_t cols, typename T>
void Helper::readInputs(std::string pathname, T (&inputs)[batch][rows][cols])
{
    float flat_matrix[batch * rows * cols];

    std::ifstream infile(pathname, std::ios::binary);
    if (!infile)
    {
        std::cout << "Error opening file!" << std::endl;
        return;
    }
    // Read dimensions
    std::cout << 1 << std::endl;
    int dim1, dim2, dim3;
    infile.read(reinterpret_cast<char *>(&dim1), sizeof(int));
    infile.read(reinterpret_cast<char *>(&dim2), sizeof(int));
    infile.read(reinterpret_cast<char *>(&dim3), sizeof(int));
    std::cout << 2 << std::endl;

    // Sanity Check
    assert(dim1 == batch);
    assert(dim2 == rows);
    assert(dim3 == cols);

    // Calculate total size
    int total_size = dim1 * dim2 * dim3;

    // Read flattened array
    infile.read(reinterpret_cast<char *>(flat_matrix), total_size * sizeof(T));
    infile.close();
    std::cout << 3 << std::endl;

    for (int i = 0; i < dim1; ++i)
    {
        for (int j = 0; j < dim2; ++j)
        {
            for (int k = 0; k < dim3; ++k)
            {
                inputs[i][j][k] = (T)flat_matrix[i * dim2 * dim3 + j * dim3 + k];
            }
        }
    }
    std::cout << 4 << std::endl;
}

/**
 * @brief Prints all the elements in a 1d array
 *
 * @tparam dim1
 * @tparam T
 * @param input
 */
template <size_t dim1, typename T>
void Helper::print(T (&input)[dim1])
{
    for (int i = 0; i < dim1; i++)
    {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
};

/**
 * @brief Returns the Argmax
 *
 * @tparam dim1
 * @tparam T
 * @param input
 */
template <size_t dim1, typename T>
int Helper::ArgMax(T (&input)[dim1])
{
    int index = -1;
    T max_var = -1;
    for (int i = 0; i < dim1; i++)
    {
        if (input[i] > max_var)
        {
            max_var = input[i];
            index = i;
        }
    }
    return index;
};

/**
 * @brief Prints all the elements in a 2d array
 *
 * @tparam dim1
 * @tparam dim2
 * @tparam T
 * @param input
 */
template <size_t dim1, size_t dim2, typename T>
void Helper::print(T (&input)[dim1][dim2])
{
    for (int i = 0; i < dim1; i++)
    {
        for (int j = 0; j < dim2; j++)
        {
            std::cout << input[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
};
/**
 * @brief Prints all the elements in a 2d array
 *
 * @tparam dim1
 * @tparam dim2
 * @tparam dim3
 * @tparam T
 * @param input
 */
template <size_t dim1, size_t dim2, size_t dim3, typename T>
void Helper::print(T (&input)[dim1][dim2][dim3])
{
    for (int i = 0; i < dim1; i++)
    {
        for (int j = 0; j < dim2; j++)
        {
            for (int k = 0; k < dim3; k++)
            {
                std::cout << input[i][j][k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
};