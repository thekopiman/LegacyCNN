#ifndef asp_h
#define asp_h

#include "conv1d.h"
#include "../utils/helper.h"
#include "../utils/matrixfunctions.h"
#include "../utils/activationfunctions.h"
#include "tdnnblock.h"
#include <string>
#include <assert.h>
#include <limits>
#include <fstream>

// We assume global_context = True and Lengths = None
template <int channels, int attention_channels, int input_width, int out_dim, typename T>
class ASP
{
public:
    ASP();
    void forward(T (&input)[channels][input_width], T (&output)[channels * 2][1]);
    ~ASP();
    void loadweights(std::string pathname);
    void loadweights(std::ifstream &infile);

private:
    TDNNBlock<1, 1, channels * 3, attention_channels, 1, input_width, input_width, 0, T> tdnn;
    Conv1d<1, 1, attention_channels, channels, 0, 1, input_width, input_width, T> conv;
    T attn[channels * 3][input_width];
    T attn1[attention_channels][input_width];
    T attn2[channels][input_width];
    T mean[channels];
    T std[channels];

    void compute_statistics(T (&x)[channels][input_width], T (&m)[channels][input_width]);
};

#include "asp.cpp"

#endif