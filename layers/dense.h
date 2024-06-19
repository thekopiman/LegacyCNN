#ifndef dense_h
#define dense_h

#include <iostream>
#include <assert.h>
#include <string>
#include <fstream>
template <int input_dim, int output_dim, typename T>
class Dense
{
public:
    Dense();
    ~Dense();

    void setWeights(T (&new_weights)[input_dim][output_dim]);

    void setBias(T (&new_bias)[output_dim]);

    // Overloading
    void setBias(std::string filename);

    // Overloading
    void setWeights(std::string filename, bool displayWeights);

    void forward(T (&input)[input_dim], T (&output)[output_dim]);

private:
    T weights[output_dim][input_dim];
    T flat_weights[output_dim * input_dim];
    T bias[output_dim];
    float flat_bias[output_dim];
};

#include "dense.cpp"

#endif