/**
 * @file batchnorm1d.h
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.2
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef batchnorm1d_h
#define batchnorm1d_h

#include <cmath>
#include <fstream>
#include <assert.h>
#include <string>
#include <iostream>

/**
 * @brief Batch Normalization 1d
 *
 * https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
 *
 * Example(s):
 *
 * BatchNorm1d<5,10,float> bn; //It's in train mode
 *
 * To construct it in eval mode make sure you do the following if you are creating it within another class
 *
 * public:
 *      MyModel<...>::MyModel(...) : bn(1) {...}
 * private:
 *      BatchNorm1d<5,10,float> bn;
 *
 *
 * Forward feed:
 * bn.forward(input, output);
 *
 *
 * @tparam channel
 * @tparam width
 * @tparam T
 */
template <int channel, int width, typename T>
class BatchNorm1d
{
public:
    BatchNorm1d();
    BatchNorm1d(int mode);
    void setGamma(T (&new_gamma)[channel]);
    void setBeta(T (&new_beta)[channel]);
    void setMean(T (&new_mean)[channel]);
    void setVar(T (&new_var)[channel]);

    // Overloading - pathname
    void setGamma(std::string pathname);
    void setBeta(std::string pathname);
    void setMean(std::string pathname);
    void setVar(std::string pathname);

    // Overloading - infile
    void setGamma(std::ifstream &infile);
    void setBeta(std::ifstream &infile);
    void setMean(std::ifstream &infile);
    void setVar(std::ifstream &infile);

    // Overloading - full

    /**
     * @brief Load the weights via pathname
     *
     * @tparam channel
     * @tparam width
     * @tparam T
     * @param pathname
     */
    void loadweights(std::string pathname);

    /**
     * @brief Load the weights via infile
     *
     * @tparam channel
     * @tparam width
     * @tparam T
     * @param infile
     */
    void loadweights(std::ifstream &infile);

    void setEps(T var);

    void forward(T (&input)[channel][width], T (&output)[channel][width]);
    ~BatchNorm1d();

private:
    T gamma[channel];
    T beta[channel];
    T mean[channel];
    T variance[channel];
    T running_mean[channel];
    T running_variance[channel];
    T eps = 1e-5;
    bool eval_mode = false;

    /**
     * @brief The `init()` function in the `BatchNorm1d` class template is initializing the parameters for each
    channel. It sets the initial values for `gamma`, `beta`, `mean`, `variance`, `running_mean`, and
    `running_variance` arrays for each channel.
     *
     * @tparam channel
     * @tparam width
     * @tparam T
     */
    void init();

    /**
     * @brief The `init_temp()` function in the `BatchNorm1d` class template is initializing the parameters for each
    channel. It sets the initial values for `mean` and `variance` arrays for each channel.
     *
     * @tparam channel
     * @tparam width
     * @tparam T
     */
    void init_temp();

    /**
     * @brief Forward feed during train mode
     *
     * @tparam channel
     * @tparam width
     * @tparam T
     * @param input
     * @param output
     */
    void forward_train(T (&input)[channel][width], T (&output)[channel][width]);
    /**
     * @brief Forward feed during eval mode
     *
     * @tparam channel
     * @tparam width
     * @tparam T
     * @param input
     * @param output
     */
    void forward_eval(T (&input)[channel][width], T (&output)[channel][width]);
};

#include "batchnorm1d.cpp"

#endif