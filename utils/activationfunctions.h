#ifndef activationfunctions_H
#define activationfunctions_H

#include <string>
#include <cmath>
#include <iostream>
#include <assert.h>
#include "matrixfunctions.h"

class ActivationFunctions
{
public:
    template <size_t rows, size_t cols, typename T>
    static void Softmax(T (&input)[rows][cols]);

    template <size_t rows, typename T>
    static void Softmax(T (&input)[rows]);

    template <size_t rows, size_t cols, typename T>
    static void ReLU(T (&input)[rows][cols]);

    template <size_t rows, size_t cols, typename T>
    static void Sigmoid(T (&input)[rows][cols]);

    template <size_t rows, typename T>
    static void Sigmoid(T (&input)[rows]);

    template <size_t rows, size_t cols, typename T>
    static void Tanh(T (&input)[rows][cols]);

    template <size_t rows, typename T>
    static void Tanh(T (&input)[rows]);
};

#include "activationfunctions.cpp"

#endif