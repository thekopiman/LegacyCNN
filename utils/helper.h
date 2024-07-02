/**
 * @file helper.h
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief Extra functions
 * @version 0.1
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef helper_h
#define helper_h

#include <string>
#include <fstream>
#include <cmath>
#include <iostream>
#include <assert.h>

/**
 * @brief Extra functions - readInputs and Print
 *
 */
class Helper
{
public:
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
    static void readInputs(std::string pathname, T (&inputs)[rows][cols]);

    /**
     * @brief Prints all the elements in a 1d array
     *
     * @tparam dim1
     * @tparam T
     * @param input
     */
    template <size_t dim1, typename T>
    static void print(T (&input)[dim1]);

    /**
     * @brief Returns the Argmax
     *
     * @tparam dim1
     * @tparam T
     * @param input
     */
    template <size_t dim1, typename T>
    static int ArgMax(T (&input)[dim1]);

    /**
     * @brief Prints all the elements in a 2d array
     *
     * @tparam dim1
     * @tparam T
     * @param input
     */
    template <size_t dim1, size_t dim2, typename T>
    static void print(T (&input)[dim1][dim2]);
};

#include "helper.cpp"

#endif