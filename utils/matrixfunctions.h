/**
 * @file matrixfunctions.h
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief Matrix Functions
 * @version 0.1
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef matrixfunctions_h
#define matrixfunctions_h

#include <string>
#include <cmath>
#include <iostream>
#include <assert.h>

/**
 * @brief MatrixFunctions include
 *
 * - Sum
 * - Flatten
 * - matrixAdd
 * - Chunk
 * - Cat
 * - Copy
 * - HadamardProduct
 * - Mean
 * - Clamp
 * - Std
 * - Reshape
 * - Norm
 * - L2Normalisation
 * - Transpose
 * - DotProduct
 * - CDist
 */
class MatrixFunctions
{
public:
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
    static T Sum(T (&input)[rows][cols]);

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
    static T Sum(T (&input)[rows]);

    /**
     * @brief Flattens a 2d Matrix into a 1d Array
     *
     * @tparam rows
     * @tparam cols
     * @tparam T
     * @param input
     * @param output
     */
    template <size_t rows, size_t cols, typename T>
    static void Flatten(T (&input)[rows][cols], T (&output)[rows * cols]);

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
    static void matrixAdd(T (&mat1)[dim1][dim2][dim3], T (&mat2)[dim1][dim2][dim3]);

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
    static void matrixAdd(T (&mat1)[dim1][dim2], T (&mat2)[dim1][dim2]);

    /**
     * @brief Matrix addition where mat1 += mat2
     *
     * @tparam dim1
     * @tparam T
     * @param mat1
     * @param mat2
     */
    template <size_t dim1, typename T>
    static void matrixAdd(T (&mat1)[dim1], T (&mat2)[dim1]);

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
    static void Chunk(T (&input)[chunk * dim1][dim2], T (&output)[chunk][dim1][dim2]);

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
    static void Cat(T (&input)[chunk][dim1][dim2], T (&output)[chunk * dim1][dim2]);

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
    static void Copy(T (&input)[dim1][dim2][dim3], T (&output)[dim1][dim2][dim3]);

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
    static void Copy(T (&input)[dim1][dim2], T (&output)[dim1][dim2]);

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
    static void Copy(T (&input)[dim1], T (&output)[dim1]);

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
    static void HadamardProduct(T (&mat1)[dim1][dim2], T (&mat2)[dim1][dim2], T (&output)[dim1][dim2]);

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
    static void HadamardProduct(T (&mat1)[dim1][dim2], T (&mat2)[dim1], T (&output)[dim1][dim2]);

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
    static void Mean(T (&input)[dim1][dim2], T (&output)[dim1]);

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
    static void Mean(T (&input)[dim1][dim2], T (&output)[dim1][dim3]);

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
    static void Clamp(T (&input)[dim1][dim2], T min, T max);

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
    static void Clamp(T (&input)[dim1][dim2], T min);

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
    static void Clamp(T (&input)[dim1], T min, T max);

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
    static T Clamp(T input, T min);

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
    static void Std(T (&input)[dim1][dim2], T (&output)[dim1]);

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
    static void Reshape(T (&input)[dim1][dim2], T (&output)[dim1]);

    /**
     * @brief Performs Norm; Default: L2 Norm
     *
     * @tparam dim1
     * @tparam T
     * @param A
     * @param B
     * @param output
     */
    template <size_t dim1, typename T>
    static T Norm(T (&A)[dim1], T (&B)[dim1]);

    /**
     * @brief Performs Norm; L_{p} Norm
     *
     * @tparam dim1
     * @tparam T
     * @param A
     * @param B
     * @param output
     * @param p
     */
    template <size_t dim1, typename T>
    static T Norm(T (&A)[dim1], T (&B)[dim1], int p);

    /**
     * @brief Performs L2Normalisation; Inplace
     *
     * @tparam dim1
     * @tparam dim2
     * @tparam T
     * @param input
     */
    template <size_t dim1, size_t dim2, typename T>
    static void L2Normalisation(T (&input)[dim1][dim2]);

    /**
     * @brief Performs Transposition
     *
     * output = input.T
     *
     * @tparam dim1
     * @tparam dim2
     * @tparam T
     * @param input
     * @param output
     */
    template <size_t dim1, size_t dim2, typename T>
    static void Transpose(T (&input)[dim1][dim2], T (&output)[dim2][dim1]);

    /**
     * @brief Performs Dot Product (2D)
     *
     * output = A dot B
     *
     * Make sure to check the shape properly
     *
     * @tparam dim1
     * @tparam dim2
     * @tparam T
     * @param A
     * @param B
     * @param output
     */
    template <size_t dim1, size_t dim2, size_t dim3, typename T>
    static void DotProduct(T (&A)[dim1][dim2], T (&B)[dim2][dim3], T (&output)[dim1][dim3]);

    /**
     * @brief Performs cdist (2D) :
     *
     * https://pytorch.org/docs/stable/generated/torch.cdist.html
     *
     * Make sure to check the shape properly
     *
     * @tparam dim1
     * @tparam dim2
     * @tparam T
     * @param A
     * @param B
     * @param output
     */
    template <size_t dim1, size_t dim2, size_t dim3, typename T>
    static void CDist(T (&A)[dim1][dim2], T (&B)[dim3][dim2], T (&output)[dim1][dim3]);

    /**
     * @brief Performs cdist (2D)
     *
     * https://pytorch.org/docs/stable/generated/torch.cdist.html
     *
     * Make sure to check the shape properly
     *
     * @tparam dim1
     * @tparam dim2
     * @tparam T
     * @param A
     * @param B
     * @param output
     * @param p
     */
    template <size_t dim1, size_t dim2, size_t dim3, typename T>
    static void CDist(T (&A)[dim1][dim2], T (&B)[dim3][dim2], T (&output)[dim1][dim3], int p);
};

#include "matrixfunctions.cpp"

#endif