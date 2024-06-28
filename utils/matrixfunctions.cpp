/**
 * @file matrixfunctions.cpp
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief Matrix Functions
 * @version 0.1
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "matrixfunctions.h"
/**
 * @brief Computes the sum of a 2d matrix and returns it.
 *
 * @tparam rows
 * @tparam cols
 * @tparam T
 * @return T
 * @param input
 */
template <size_t rows, size_t cols, typename T>
T MatrixFunctions::Sum(T (&input)[rows][cols])
{
    T total = 0.0f;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            total += input[i][j];
        }
    }
    return total;
};

/**
 * @brief Computes the sum of a 1d array and returns it.
 *
 * @tparam rows
 * @tparam cols
 * @tparam T
 * @return T
 * @param input
 */
template <size_t rows, typename T>
T MatrixFunctions::Sum(T (&input)[rows])
{
    T total = 0.0;
    for (int i = 0; i < rows; i++)
    {
        total += input[i];
    }
    return total;
};

/**
 * @brief Flattens a 2d Matrix into a 1d Array
 *
 * @tparam rows
 * @tparam cols
 * @tparam T
 */
template <size_t rows, size_t cols, typename T>
void MatrixFunctions::Flatten(T (&input)[rows][cols], T (&output)[rows * cols])
{
    int index = 0;
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            output[index++] = input[i][j];
        }
    }
};

/**
 * @brief Matrix addition where mat1 += mat2
 *
 * @tparam dim1
 * @tparam dim2
 * @tparam dim3
 * @tparam T
 * @param mat1
 * @param mat2
 */
template <size_t dim1, size_t dim2, size_t dim3, typename T>
void MatrixFunctions::matrixAdd(T (&mat1)[dim1][dim2][dim3], T (&mat2)[dim1][dim2][dim3])
{
    for (int i = 0; i < dim1; i++)
    {
        for (int j = 0; j < dim2; j++)
        {
            for (int k = 0; k < dim3; k++)
            {
                mat1[i][j][k] += mat2[i][j][k];
            }
        }
    }
};

/**
 * @brief Matrix addition where mat1 += mat2
 *
 * @tparam dim1
 * @tparam dim2
 * @tparam T
 * @param mat1
 * @param mat2
 */
template <size_t dim1, size_t dim2, typename T>
void MatrixFunctions::matrixAdd(T (&mat1)[dim1][dim2], T (&mat2)[dim1][dim2])
{
    for (int i = 0; i < dim1; i++)
    {
        for (int j = 0; j < dim2; j++)
        {
            mat1[i][j] += mat2[i][j];
        }
    }
};

/**
 * @brief Matrix addition where mat1 += mat2
 *
 * @tparam dim1
 * @tparam T
 * @param mat1
 * @param mat2
 */
template <size_t dim1, typename T>
void MatrixFunctions::matrixAdd(T (&mat1)[dim1], T (&mat2)[dim1])
{
    for (int i = 0; i < dim1; i++)
    {
        mat1[i] += mat2[i];
    }
};

/**
 * @brief Chunk an input of size [chunk * dim1][dim2] into output of size [chunk][dim1][dim2]
 * https://pytorch.org/docs/stable/generated/torch.chunk.html
 *
 * @tparam dim1
 * @tparam dim2
 * @tparam chunk
 * @tparam T
 * @param input
 * @param output
 */
template <size_t dim1, size_t dim2, size_t chunk, typename T>
void MatrixFunctions::Chunk(T (&input)[chunk * dim1][dim2], T (&output)[chunk][dim1][dim2])
{
    for (int i = 0; i < chunk; i++)
    {
        for (int j = 0; j < dim1; j++)
        {
            for (int k = 0; k < dim2; k++)
            {
                output[i][j][k] = input[i * dim1 + j][k];
                // std::cout << i << j << " " << i * dim1 + j << " " << output[i][j][k] << std::endl;
            }
        }
    }
};

/**
 * @brief Catenate an input of size [chunk][dim1][dim2] into output of size [chunk * dim1][dim2]
 *
 * @tparam dim1
 * @tparam dim2
 * @tparam chunk
 * @tparam T
 * @param input
 * @param output
 */
template <size_t dim1, size_t dim2, size_t chunk, typename T>
void MatrixFunctions::Cat(T (&input)[chunk][dim1][dim2], T (&output)[chunk * dim1][dim2])
{
    for (int i = 0; i < chunk; i++)
    {
        for (int j = 0; j < dim1; j++)
        {
            for (int k = 0; k < dim2; k++)
            {
                output[i * dim1 + j][k] = input[i][j][k];
            }
        }
    }
};

/**
 * @brief Deep Copy where
 *
 * output = input;
 *
 * @tparam dim1
 * @tparam dim2
 * @tparam dim3
 * @tparam T
 * @param input
 * @param output
 */
template <size_t dim1, size_t dim2, size_t dim3, typename T>
void MatrixFunctions::Copy(T (&input)[dim1][dim2][dim3], T (&output)[dim1][dim2][dim3])
{
    for (int i = 0; i < dim1; i++)
    {
        for (int j = 0; j < dim2; j++)
        {
            for (int k = 0; k < dim3; k++)
            {
                output[i][j][k] = input[i][j][k];
            }
        }
    }
};

/**
 * @brief Deep Copy where
 *
 * output = input;
 *
 * @tparam dim1
 * @tparam dim2
 * @tparam T
 * @param input
 * @param output
 */
template <size_t dim1, size_t dim2, typename T>
void MatrixFunctions::Copy(T (&input)[dim1][dim2], T (&output)[dim1][dim2])
{
    for (int i = 0; i < dim1; i++)
    {
        for (int j = 0; j < dim2; j++)
        {
            output[i][j] = input[i][j];
        }
    }
};

/**
 * @brief Deep Copy where
 *
 * output = input;
 *
 * @tparam dim1
 * @tparam T
 * @param input
 * @param output
 */
template <size_t dim1, typename T>
void MatrixFunctions::Copy(T (&input)[dim1], T (&output)[dim1])
{
    for (int i = 0; i < dim1; i++)
    {
        output[i] = input[i];
    }
};

/**
 * @brief Perform PyTorch default "Product" between 2 matrices. A * B
 * https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
 *
 * Here dim(A) = dim(B)
 *
 * @tparam dim1
 * @tparam dim2
 * @tparam T
 * @param mat1
 * @param mat2
 * @param output
 */
template <size_t dim1, size_t dim2, typename T>
void MatrixFunctions::HadamardProduct(T (&mat1)[dim1][dim2], T (&mat2)[dim1][dim2], T (&output)[dim1][dim2])
{
    for (int i = 0; i < dim1; i++)
    {
        for (int j = 0; j < dim2; j++)
        {
            output[i][j] = mat1[i][j] * mat2[i][j];
        }
    }
}

/**
 * @brief Perform PyTorch default "Product" between 2 matrices. A * B
 *
 * https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
 *
 * Broadcasting feature is applied on mat2.
 * https://numpy.org/doc/stable/user/basics.broadcasting.html
 *
 * @tparam dim1
 * @tparam dim2
 * @tparam T
 * @param mat1
 * @param mat2
 * @param output
 */
template <size_t dim1, size_t dim2, typename T>
void MatrixFunctions::HadamardProduct(T (&mat1)[dim1][dim2], T (&mat2)[dim1], T (&output)[dim1][dim2])
{

    for (int i = 0; i < dim1; i++)
    {
        for (int j = 0; j < dim2; j++)
        {
            output[i][j] = mat1[i][j] * mat2[i];
        }
    }
}

/**
 * @brief Computes the mean on dim2. Here output has a shape of (dim1)
 *
 * @tparam dim1
 * @tparam dim2
 * @tparam T
 * @param input
 * @param output
 */
template <size_t dim1, size_t dim2, typename T>
void MatrixFunctions::Mean(T (&input)[dim1][dim2], T (&output)[dim1])
{
    for (int i = 0; i < dim1; i++)
    {
        output[i] = Sum(input[i]) / dim2;
    }
}

/**
 * @brief Computes the mean on dim2. Here output has a shape of (dim1, 1)
 * where dim3 = 1
 *
 * @tparam dim1
 * @tparam dim2
 * @tparam dim3
 * @tparam T
 * @param input
 * @param output
 */
template <size_t dim1, size_t dim2, size_t dim3, typename T>
void MatrixFunctions::Mean(T (&input)[dim1][dim2], T (&output)[dim1][dim3])
{
    assert(dim3 == 1);
    for (int i = 0; i < dim1; i++)
    {
        output[i][0] = Sum(input[i]) / dim2;
    }
}

/**
 * @brief yi​=min(max(xi​,min_valuei​),max_valuei​)
 *
 * https://pytorch.org/docs/stable/generated/torch.clamp.html
 *
 * Here, it clamps both min and max
 *
 * @tparam dim1
 * @tparam dim2
 * @tparam T
 * @param input
 * @param min
 * @param max
 */
template <size_t dim1, size_t dim2, typename T>
void MatrixFunctions::Clamp(T (&input)[dim1][dim2], T min, T max)
{
    for (int i = 0; i < dim1; i++)
    {
        for (int j = 0; j < dim2; j++)
        {
            if (input[i][j] < min)
            {
                input[i][j] = min;
            }
            else if (input[i][j] > max)
            {
                input[i][j] = max;
            }
        }
    }
}

/**
 * @brief yi​=max(xi​,min_valuei​)
 *
 * https://pytorch.org/docs/stable/generated/torch.clamp.html
 *
 * Here, it clamps min only
 *
 * @tparam dim1
 * @tparam dim2
 * @tparam T
 * @param input
 * @param min
 * @param max
 */
template <size_t dim1, size_t dim2, typename T>
void MatrixFunctions::Clamp(T (&input)[dim1][dim2], T min)
{
    for (int i = 0; i < dim1; i++)
    {
        for (int j = 0; j < dim2; j++)
        {
            if (input[i][j] < min)
            {
                input[i][j] = min;
            }
        }
    }
}

/**
 * @brief yi​=min(max(xi​,min_valuei​),max_valuei​)
 *
 * https://pytorch.org/docs/stable/generated/torch.clamp.html
 *
 * Here, it clamps both min and max
 *
 * @tparam dim1
 * @tparam T
 * @param input
 * @param min
 * @param max
 */
template <size_t dim1, typename T>
void MatrixFunctions::Clamp(T (&input)[dim1], T min, T max)
{
    for (int i = 0; i < dim1; i++)
    {
        if (input[i] < min)
        {
            input[i] = min;
        }
        else if (input[i] > max)
        {
            input[i] = max;
        }
    }
}

/**
 * @brief y​=max(x​,min​)
 *
 * https://pytorch.org/docs/stable/generated/torch.clamp.html
 *
 * Here, it clamps min only
 *
 * @tparam T
 * @param input
 * @param min
 * @return T
 */
template <typename T>
T MatrixFunctions::Clamp(T input, T min)
{
    if (input < min)
    {
        return min;
    }
    else
    {
        return input;
    }
}

/**
 * @brief Computes the population std of input by channel
 *
 * such that Y = max(Var(input_i), 1e-12)
 * with a clamp of min 1e-12.
 *
 * @tparam dim1
 * @tparam dim2
 * @tparam T
 * @param input
 * @param output
 */
template <size_t dim1, size_t dim2, typename T>
void MatrixFunctions::Std(T (&input)[dim1][dim2], T (&output)[dim1])
{
    T eps = 1e-12;
    T mean[dim1];
    T variance[dim1];
    for (int i = 0; i < dim1; i++)
    {
        mean[i] = 0;
        variance[i] = 0;
    }

    // Calculate mean E(X) first
    for (int c = 0; c < dim1; c++)
    {
        for (int i = 0; i < dim2; i++)
        {
            mean[c] += input[c][i];
        }
        mean[c] /= dim2;
    }

    // Calculate Population Variance Var(X)
    for (int c = 0; c < dim1; c++)
    {
        for (int i = 0; i < dim2; i++)
        {
            variance[c] += (input[c][i] - mean[c]) * (input[c][i] - mean[c]);
        }
        variance[c] /= dim2;
        output[c] = std::sqrt(Clamp(variance[c], eps));
    }
}

/**
 * @brief Reshapes a input of size (dim1, 1) into output (dim1)
 *
 * @tparam dim1
 * @tparam dim2
 * @tparam T
 * @param input
 * @param output
 */
template <size_t dim1, size_t dim2, typename T>
void MatrixFunctions::Reshape(T (&input)[dim1][dim2], T (&output)[dim1])
{
    assert(dim2 == 1);
    for (int i = 0; i < dim1; i++)
    {
        output[i] = input[i][0];
    }
};