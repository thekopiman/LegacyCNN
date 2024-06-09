#ifndef softmax_H
#define softmax_H

#include <string>
#include <fstream>
#include <cmath>
#include <iostream>

class Helper
{
public:
    template <size_t rows, size_t cols, typename T>
    static void Softmax(T (&input)[rows][cols])
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                input[i][j] = (T)exp((double)input[i][j]);
            }
        }

        T sumAll = Sum(input);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                input[i][j] /= sumAll;
            }
        }
    };
    template <size_t rows, typename T>
    static void Softmax(T (&input)[rows])
    {
        for (int i = 0; i < rows; i++)
        {
            input[i] = (T)exp((double)input[i]);
        }

        float sumAll = Sum(input);

        for (int i = 0; i < rows; i++)
        {
            input[i] /= sumAll;
        }
    };

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
    static T Flatten(T (&input)[rows][cols], T (&output)[rows * cols])
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

    template <size_t rows, size_t cols, typename T>
    static void ReLU(T (&input)[rows][cols])
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (input[i][j] <= 0)
                {
                    input[i][j] = 0;
                }
            }
        }
    }

    template <size_t rows, size_t cols, typename T>
    static void readInputs(std::string filename, T (&inputs)[rows][cols])
    {
        float flat_matrix[rows * cols];

        std::ifstream infile(filename, std::ios::binary);
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
        infile.read(reinterpret_cast<char *>(flat_matrix), total_size * sizeof(float));
        infile.close();

        for (int i = 0; i < dim1; ++i)
        {
            for (int j = 0; j < dim2; ++j)
            {
                inputs[i][j] = (T)flat_matrix[i * dim2 + j];
            }
        }
    }
};

#endif