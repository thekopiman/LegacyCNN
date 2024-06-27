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
    BatchNorm1d(int mode);
    void setGamma(T (&new_gamma)[channel]);
    void setBeta(T (&new_beta)[channel]);
    void setMean(T (&new_mean)[channel]);
    void setVar(T (&new_var)[channel]);

    // Overloading - pathname
    void setGamma(std::string filename);
    void setBeta(std::string filename);
    void setMean(std::string filename);
    void setVar(std::string filename);

    // Overloading - infile
    void setGamma(std::ifstream &infile);
    void setBeta(std::ifstream &infile);
    void setMean(std::ifstream &infile);
    void setVar(std::ifstream &infile);

    // Overloading - full
    void loadweights(std::string filename);
    void loadweights(std::ifstream &infile);

    void setEps(T var);
    void forward_train(T (&input)[channel][width], T (&output)[channel][width]);
    void forward_eval(T (&input)[channel][width], T (&output)[channel][width]);
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

    void init();
};

#include "batchnorm1d.cpp"

#endif