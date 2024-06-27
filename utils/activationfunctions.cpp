/**
 * @file activationfunctions.cpp
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief Consist of Activation functions used in ML
 * @version 0.1
 * @date 2024-06-27
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "activationfunctions.h"

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
void ActivationFunctions::Softmax(T (&input)[rows][cols])
{
    for (int i = 0; i < rows; i++)
    {
        Softmax(input[i]);
    }
};

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
void ActivationFunctions::Softmax(T (&input)[rows])
{
    for (int i = 0; i < rows; i++)
    {
        input[i] = (T)exp((double)input[i]); // Need to up/downcast to double for exp. Subsequently, it will be down/up cast
    }

    T sumAll = MatrixFunctions::Sum(input);

    for (int i = 0; i < rows; i++)
    {
        input[i] /= sumAll;
    }
};

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
void ActivationFunctions::ReLU(T (&input)[rows][cols])
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
void ActivationFunctions::Sigmoid(T (&input)[rows][cols])
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            input[i][j] = 1 / (1 + (T)exp(-(double)input[i][j]));
        }
    }
}

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
void ActivationFunctions::Sigmoid(T (&input)[rows])
{
    for (int i = 0; i < rows; i++)
    {
        input[i] = 1 / (1 + (T)exp(-(double)input[i]));
    }
}

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
void ActivationFunctions::Tanh(T (&input)[rows][cols])
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            input[i][j] = (exp(input[i][j]) - exp(-input[i][j])) / (exp(input[i][j]) + exp(-input[i][j]));
        }
    }
}

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
void ActivationFunctions::Tanh(T (&input)[rows])
{
    for (int i = 0; i < rows; i++)
    {
        input[i] = (exp(input[i]) - exp(-input[i])) / (exp(input[i]) + exp(-input[i]));
    }
}