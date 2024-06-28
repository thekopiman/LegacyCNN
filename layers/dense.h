/**
 * @file dense.cpp
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.1
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef dense_h
#define dense_h

#include <iostream>
#include <assert.h>
#include <string>
#include <fstream>

/**
 * @brief Dense/Linear Layer
 *
 * https://pytorch.org/docs/stable/generated/torch.nn.Linear.htmls
 *
 * @tparam input_dim
 * @tparam output_dim
 * @tparam T
 */
template <int input_dim, int output_dim, typename T>
class Dense
{
public:
    /**
     * @brief Construct a new Dense<input_dim, output_dim, T>::Dense object
     *
     * @tparam input_dim
     * @tparam output_dim
     * @tparam T
     */
    Dense();
    ~Dense();

    void setWeights(T (&new_weights)[input_dim][output_dim]);

    void setBias(T (&new_bias)[output_dim]);

    // Overloading - pathname
    void setBias(std::string pathname);
    void setWeights(std::string pathname, bool displayWeights);

    // Overloading - infile

    void setBias(std::ifstream &infile);
    void setWeights(std::ifstream &infile, bool displayWeights);

    // Overloading - full

    /**
     * @brief Load weights via infile
     *
     * @tparam input_dim
     * @tparam output_dim
     * @tparam T
     * @param infile
     */
    void loadweights(std::ifstream &infile);

    /**
     * @brief Load weights via pathname
     *
     * @tparam input_dim
     * @tparam output_dim
     * @tparam T
     * @param pathname
     */
    void loadweights(std::string pathname);

    /**
     * @brief Performs forward feed
     *
     * @tparam input_dim
     * @tparam output_dim
     * @tparam T
     * @param input
     * @param output
     */
    void forward(T (&input)[input_dim], T (&output)[output_dim]);

private:
    T weights[output_dim][input_dim];
    T flat_weights[output_dim * input_dim];
    T bias[output_dim];
    float flat_bias[output_dim];
};

#include "dense.cpp"

#endif