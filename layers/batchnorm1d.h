#ifndef batchnorm1d_h
#define batchnorm1d_h

#include <cmath>
#include <fstream>
#include <assert.h>
#include <string>
#include <iostream>

// https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
template <int channel, int width, typename T>
class BatchNorm1d
{
public:
    BatchNorm1d();
    void setGamma(T (&new_gamma)[channel]);
    void setBeta(T (&new_beta)[channel]);

    // Overloading
    void setGamma(std::string filename);

    // Overloading
    void setBeta(std::string filename);

    void setEps(T var);
    void forward(T (&input)[channel][width], T (&output)[channel][width]);
    ~BatchNorm1d();

private:
    T gamma[channel];
    T beta[channel];
    T mean[channel];
    T variance[channel];
    T eps = 1e-5;
};

#include "batchnorm1d.cpp"

#endif