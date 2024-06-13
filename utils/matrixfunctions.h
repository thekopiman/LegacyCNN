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
    static T Sum(T (&input)[rows][cols])
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

    template <size_t rows, typename T>
    static T Sum(T (&input)[rows])
    {
        T total = 0.0;
        for (int i = 0; i < rows; i++)
        {
            total += input[i];
        }
        return total;
    };

    template <size_t rows, size_t cols, typename T>
    static void Flatten(T (&input)[rows][cols], T (&output)[rows * cols])
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
    // mat1 += mat2
    // The output will be on mat1
    template <size_t dim1, size_t dim2, size_t dim3, typename T>
    static void matrixAdd(T (&mat1)[dim1][dim2][dim3], T (&mat2)[dim1][dim2][dim3])
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

    // mat1 += mat2
    // The output will be on mat1
    template <size_t dim1, size_t dim2, typename T>
    static void matrixAdd(T (&mat1)[dim1][dim2], T (&mat2)[dim1][dim2])
    {
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                mat1[i][j] += mat2[i][j];
            }
        }
    };

    // Chunk is not necessary here as we are spliting based on channel/dim1
};

#endif