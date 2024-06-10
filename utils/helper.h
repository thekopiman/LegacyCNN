#ifndef helper_h
#define helper_h

#include <string>
#include <fstream>
#include <cmath>
#include <iostream>
#include <assert.h>

class Helper
{
public:
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
};

#endif