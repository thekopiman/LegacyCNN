#include "activationfunctions.h"

template <size_t rows, size_t cols, typename T>
void ActivationFunctions::Softmax(T (&input)[rows][cols])
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            input[i][j] = (T)exp((double)input[i][j]); // Need to up/downcast to double for exp. Subsequently, it will be down/up cast
        }
    }

    T sumAll = MatrixFunctions::Sum(input);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            input[i][j] /= sumAll;
        }
    }
};

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

template <size_t rows, typename T>
void ActivationFunctions::Sigmoid(T (&input)[rows])
{
    for (int i = 0; i < rows; i++)
    {
        input[i] = 1 / (1 + (T)exp(-(double)input[i]));
    }
}

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

template <size_t rows, typename T>
void ActivationFunctions::Tanh(T (&input)[rows])
{
    for (int i = 0; i < rows; i++)
    {
        input[i] = (exp(input[i]) - exp(-input[i])) / (exp(input[i]) + exp(-input[i]));
    }
}