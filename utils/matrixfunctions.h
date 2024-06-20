#ifndef matrixfunctions_h
#define matrixfunctions_h

#include <string>
#include <cmath>
#include <iostream>
#include <assert.h>

class MatrixFunctions
{
public:
    template <size_t rows, size_t cols, typename T>
    static T Sum(T (&input)[rows][cols]);

    template <size_t rows, typename T>
    static T Sum(T (&input)[rows]);

    template <size_t rows, size_t cols, typename T>
    static void Flatten(T (&input)[rows][cols], T (&output)[rows * cols]);

    // mat1 += mat2
    // The output will be on mat1
    template <size_t dim1, size_t dim2, size_t dim3, typename T>
    static void matrixAdd(T (&mat1)[dim1][dim2][dim3], T (&mat2)[dim1][dim2][dim3]);
    template <size_t dim1, size_t dim2, typename T>
    static void matrixAdd(T (&mat1)[dim1][dim2], T (&mat2)[dim1][dim2]);
    template <size_t dim1, typename T>
    static void matrixAdd(T (&mat1)[dim1], T (&mat2)[dim1]);

    // Chunk
    template <size_t dim1, size_t dim2, size_t chunk, typename T>
    static void Chunk(T (&input)[chunk * dim1][dim2], T (&output)[chunk][dim1][dim2]);

    // Cat
    template <size_t dim1, size_t dim2, size_t chunk, typename T>
    static void Cat(T (&input)[chunk][dim1][dim2], T (&output)[chunk * dim1][dim2]);

    // Copy
    template <size_t dim1, size_t dim2, size_t dim3, typename T>
    static void Copy(T (&input)[dim1][dim2][dim3], T (&output)[dim1][dim2][dim3]);

    template <size_t dim1, size_t dim2, typename T>
    static void Copy(T (&input)[dim1][dim2], T (&output)[dim1][dim2]);

    template <size_t dim1, typename T>
    static void Copy(T (&input)[dim1], T (&output)[dim1]);

    // HadamardProduct
    // (mat1*mat2)[i][j] = mat1[i][j] * mat2[i][j]

    template <size_t dim1, size_t dim2, typename T>
    static void HadamardProduct(T (&mat1)[dim1][dim2], T (&mat2)[dim1][dim2], T (&output)[dim1][dim2]);

    // Hadamard Product here aim to replicate the Broadcasting feature in PyTorch
    template <size_t dim1, size_t dim2, typename T>
    static void HadamardProduct(T (&mat1)[dim1][dim2], T (&mat2)[dim1], T (&output)[dim1][dim2]);

    // Mean will be based on dim2
    template <size_t dim1, size_t dim2, typename T>
    static void Mean(T (&input)[dim1][dim2], T (&output)[dim1]);
    template <size_t dim1, size_t dim2, size_t dim3, typename T>
    static void Mean(T (&input)[dim1][dim2], T (&output)[dim1][dim3]);

    // Clamp
    template <size_t dim1, size_t dim2, typename T>
    static void Clamp(T (&input)[dim1][dim2], T min, T max);

    // Clamp - min only
    template <size_t dim1, size_t dim2, typename T>
    static void Clamp(T (&input)[dim1][dim2], T min);

    // Clamp
    template <size_t dim1, typename T>
    static void Clamp(T (&input)[dim1], T min, T max);

    // Clamp - min only
    template <typename T>
    static T Clamp(T input, T min);

    // Obtain std
    template <size_t dim1, size_t dim2, typename T>
    static void Std(T (&input)[dim1][dim2], T (&output)[dim1]);

    // Reshape
    template <size_t dim1, size_t dim2, typename T>
    static void Reshape(T (&input)[dim1][dim2], T (&output)[dim1]);
};

#include "matrixfunctions.cpp"

#endif