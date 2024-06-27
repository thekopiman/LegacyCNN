/**
 * @file activationfunctions.h
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief Consist of Activation functions used in ML
 * @version 0.1
 * @date 2024-06-27
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef activationfunctions_H
#define activationfunctions_H

#include <string>
#include <cmath>
#include <iostream>
#include <assert.h>
#include "matrixfunctions.h"

/**
 * @brief Consist of ActivationFunctions
 *
 */
class ActivationFunctions
{
public:
    /**
     * @brief Perform Softmax
     * It will edit the pointer content directly.
     *
     * // Example:
     * ActivationFunctions::Softmax(input)
     *
     * OR
     *
     * ActivationFunctions::Softmax<5, 5, float>(input)
     *
     * @tparam rows
     * @tparam cols
     * @tparam T
     * @param input
     */
    template <size_t rows, size_t cols, typename T>
    static void Softmax(T (&input)[rows][cols]);

    /**
     * @brief Perform Softmax
     * It will edit the pointer content directly.
     *
     * // Example:
     * ActivationFunctions::Softmax(input)
     *
     * OR
     *
     * ActivationFunctions::Softmax<5,float>(input)
     *
     * @tparam rows
     * @tparam T
     * @param input
     */
    template <size_t rows, typename T>
    static void Softmax(T (&input)[rows]);

    /**
     * @brief Perform ReLU
     * It will edit the pointer content directly.
     *
     * // Example:
     * ActivationFunctions::ReLU(input)
     *
     * OR
     *
     * ActivationFunctions::ReLU<5, 5, float>(input)
     *
     * @tparam rows
     * @tparam cols
     * @tparam T
     * @param input
     */
    template <size_t rows, size_t cols, typename T>
    static void ReLU(T (&input)[rows][cols]);

    /**
     * @brief Perform Sigmoid.
     * It will edit the pointer content directly.
     *
     * // Example:
     * ActivationFunctions::Sigmoid(input)
     *
     * OR
     *
     * ActivationFunctions::Sigmoid<5, 5, float>(input)
     *
     * @tparam rows
     * @tparam cols
     * @tparam T
     * @param input
     */
    template <size_t rows, size_t cols, typename T>
    static void Sigmoid(T (&input)[rows][cols]);

    /**
     * @brief Perform Sigmoid
     * It will edit the pointer content directly.
     *
     * // Example:
     * ActivationFunctions::Sigmoid(input)
     *
     * OR
     *
     * ActivationFunctions::Sigmoid<5, float>(input)
     *
     * @tparam rows
     * @tparam T
     * @param input
     */
    template <size_t rows, typename T>
    static void Sigmoid(T (&input)[rows]);

    /**
     * @brief Perform Tanh
     * It will edit the pointer content directly.
     *
     * // Example:
     * ActivationFunctions::Tanh(input)
     *
     * OR
     *
     * ActivationFunctions::Tanh<5, 5, float>(input)
     *
     * @tparam rows
     * @tparam cols
     * @tparam T
     * @param input
     */
    template <size_t rows, size_t cols, typename T>
    static void Tanh(T (&input)[rows][cols]);

    /**
     * @brief Perform Tanh
     * It will edit the pointer content directly.
     *
     * // Example:
     * ActivationFunctions::Tanh(input)
     *
     * OR
     *
     * ActivationFunctions::Tanh<5, float>(input)
     *
     * @tparam rows
     * @tparam T
     * @param input
     */
    template <size_t rows, typename T>
    static void Tanh(T (&input)[rows]);
};

#include "activationfunctions.cpp"

#endif