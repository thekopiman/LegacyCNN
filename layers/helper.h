#ifndef softmax_H
#define softmax_H

#include <cmath>
#include <iostream>

class Helper
{
public:
    template <size_t rows, size_t cols>
    static void Softmax(float (&input)[rows][cols])
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                input[i][j] = (float)exp((double)input[i][j]);
            }
        }

        float sumAll = Sum(input);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                input[i][j] /= sumAll;
            }
        }
    };

    template <size_t rows, size_t cols>
    static float Sum(float (&input)[rows][cols])
    {
        float total = 0.0f;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                total += input[i][j];
            }
        }
        return total;
    };

    template <size_t rows, size_t cols>
    static float Flatten(float (&input)[rows][cols], float (&output)[rows * cols])
    {
        int index = 0;
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                output[index++] = input[i][j];
            }
        }
    }
};

#endif