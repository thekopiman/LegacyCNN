/**
 * @file asp.cpp
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.3
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "asp.h"

/**
 * @brief Construct a new asp<channels, attention channels, input width, out dim, t>::asp object
 *
 * @tparam channels
 * @tparam attention_channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 */
template <int channels, int attention_channels, int input_width, int out_width, typename T>
ASP<channels, attention_channels, input_width, out_width, T>::ASP(){};

/**
 * @brief Destroy the asp<channels, attention channels, input width, out dim, t>::asp object
 *
 * @tparam channels
 * @tparam attention_channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 */
template <int channels, int attention_channels, int input_width, int out_width, typename T>
ASP<channels, attention_channels, input_width, out_width, T>::~ASP(){};

/**
 * @brief Computes the forward feed
 *
 * @tparam channels
 * @tparam attention_channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 * @param input
 * @param output
 */
template <int channels, int attention_channels, int input_width, int out_width, typename T>
void ASP<channels, attention_channels, input_width, out_width, T>::forward(T (&input)[channels][input_width], T (&output)[channels * 2][1])
{
    MatrixFunctions::Mean(input, this->mean);
    MatrixFunctions::Std(input, this->std);

    // attn = torch.cat([x, mean, std], dim=1)
    for (int i = 0; i < channels; i++)
    {
        for (int j = 0; j < input_width; j++)
        {
            this->attn[i][j] = input[i][j];
            this->attn[channels + i][j] = this->mean[i];    // mean = mean.unsqueeze(2).repeat(1, 1, L)
            this->attn[2 * channels + i][j] = this->std[i]; // std = std.unsqueeze(2).repeat(1, 1, L)
        }
    }

    // Helper::print(this->attn);

    // apply layers
    this->tdnn.forward(this->attn, this->attn1);
    ActivationFunctions::Tanh(this->attn1);
    this->conv.forward(this->attn1, this->attn2);

    // Filter out zero-paddings is redundant here

    // Helper::print(this->attn2);

    ActivationFunctions::Softmax(this->attn2);

    // Helper::print(this->attn2);

    // Obtain Pooled statistics
    compute_statistics(input, attn2);

    for (int i = 0; i < channels; i++)
    {
        output[i][0] = this->mean[i];
        output[channels + i][0] = this->std[i];
    }
};

/**
 * @brief Computes the forward feed
 *
 * Since Batch = 1, dim(lengths) == 1
 *
 * @tparam channels
 * @tparam attention_channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 * @param input
 * @param lengths
 * @param output
 */
template <int channels, int attention_channels, int input_width, int out_width, typename T>
void ASP<channels, attention_channels, input_width, out_width, T>::forward(T (&input)[channels][input_width], T &lengths, T (&output)[channels * 2][1])
{
    T tempLengths[channels];
    assert(lengths > 0 && "Make sure that lengths contain at least a Natural number to prevent division by 0.");

    for (int i = 0; i < channels; i++)
    {
        tempLengths[i] = lengths;
    }

    forward(input, tempLengths, output);
}

/**
 * @brief Computes the forward feed
 *
 * Since Batch = 1, dim(lengths) == 1
 *
 * @tparam channels
 * @tparam attention_channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 * @param input
 * @param lengths
 * @param output
 */
template <int channels, int attention_channels, int input_width, int out_width, typename T>
void ASP<channels, attention_channels, input_width, out_width, T>::forward(T (&input)[channels][input_width], T (&lengths)[channels], T (&output)[channels * 2][1])
{
    resetMask();

    bool ExistNonZero = false;
    for (int i = 0; i < channels; i++)
    {
        if (lengths[i] > 0)
        {
            ExistNonZero = true;
        }
    }

    assert(ExistNonZero && "Make sure that lengths contain at least a Natural number to prevent division by 0.");

    // Mask here has a dim of [N, C, L] instead of the one [N, 1, L]. Where N = 1
    // This is because Hadamard Product in MatrixFunctions requires same dimensions.

    // Length to mask
    // https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/dataio/dataio.html#length_to_mask

    for (int i = 0; i < channels; i++)
    {
        for (int j = 0; j < std::ceil(lengths[i] * input_width) && j < input_width; j++)
        {
            this->mask[i][j] = 1;
        }
        this->total[i] = MatrixFunctions::Sum(this->mask[i]);
    }

    // mask / total

    for (int i = 0; i < channels; i++)
    {
        for (int j = 0; j < input_width; j++)
        {
            this->mask[i][j] /= this->total[i];
        }
    }

    // mean, std = _compute_statistics(x, mask / total)
    compute_statistics(input, this->mask);

    // attn = torch.cat([x, mean, std], dim=1)
    for (int i = 0; i < channels; i++)
    {
        for (int j = 0; j < input_width; j++)
        {
            this->attn[i][j] = input[i][j];
            this->attn[channels + i][j] = this->mean[i];    // mean = mean.unsqueeze(2).repeat(1, 1, L)
            this->attn[2 * channels + i][j] = this->std[i]; // std = std.unsqueeze(2).repeat(1, 1, L)
        }
    }

    // Helper::print(this->attn);

    // apply layers
    this->tdnn.forward(this->attn, this->attn1);
    ActivationFunctions::Tanh(this->attn1);
    this->conv.forward(this->attn1, this->attn2);

    // Filter out zero-paddings
    for (int j = 0; j < input_width; j++)
    {
        if (this->mask[0][j] == 0)
        {
            for (int i = 0; i < channels; i++)
            {
                this->attn2[i][j] = std::numeric_limits<T>::lowest();
            }
        }
    }

    // Helper::print(this->attn2);

    ActivationFunctions::Softmax(this->attn2);

    // Helper::print(this->attn2);

    // Obtain Pooled statistics
    compute_statistics(input, attn2);

    for (int i = 0; i < channels; i++)
    {
        output[i][0] = this->mean[i];
        output[channels + i][0] = this->std[i];
    }
}

/**
 * @brief Resets the Mask to 0
 *
 * @tparam channels
 * @tparam attention_channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 */
template <int channels, int attention_channels, int input_width, int out_width, typename T>
void ASP<channels, attention_channels, input_width, out_width, T>::resetMask()
{
    for (int i = 0; i < channels; i++)
    {
        for (int j = 0; j < input_width; j++)
        {
            this->mask[i][j] = 0;
        }
        this->total[i] = 0;
    }
}

/**
 * @brief Compute Statistics
 *
 * @tparam channels
 * @tparam attention_channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 * @param input
 * @param output
 */
template <int channels, int attention_channels, int input_width, int out_width, typename T>
void ASP<channels, attention_channels, input_width, out_width, T>::compute_statistics(T (&x)[channels][input_width], T (&m)[channels][input_width])
{
    T temp[channels][input_width];

    // Obtain Mean here
    MatrixFunctions::HadamardProduct(x, m, temp);
    for (int i = 0; i < channels; i++)
    {
        this->mean[i] = MatrixFunctions::Sum(temp[i]);
    }

    // Obtain Population Std here

    for (int i = 0; i < channels; i++)
    {
        for (int j = 0; j < input_width; j++)
        {
            x[i][j] -= this->mean[i]; // x - mean
            x[i][j] *= x[i][j];       // pow(2)
        }
    }

    MatrixFunctions::HadamardProduct(x, m, temp);
    for (int i = 0; i < channels; i++)
    {
        this->std[i] = MatrixFunctions::Clamp(MatrixFunctions::Sum(temp[i]), (T)1e-12);
        this->std[i] = std::sqrt(this->std[i]);
    }
}

/**
 * @brief Loadweights via pathname
 *
 * @tparam channels
 * @tparam attention_channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 * @param pathname
 */
template <int channels, int attention_channels, int input_width, int out_width, typename T>
void ASP<channels, attention_channels, input_width, out_width, T>::loadweights(std::string pathname)
{
    std::ifstream infile(pathname, std::ios::binary);
    if (!infile)
    {
        std::cout << "Error opening file!" << std::endl;
        return;
    }

    tdnn.loadweights(infile);
    conv.loadweights(infile);

    infile.close();
};

/**
 * @brief Loadweights via infile
 *
 * @tparam channels
 * @tparam attention_channels
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 * @param infile
 */
template <int channels, int attention_channels, int input_width, int out_width, typename T>
void ASP<channels, attention_channels, input_width, out_width, T>::loadweights(std::ifstream &infile)
{
    tdnn.loadweights(infile);
    conv.loadweights(infile);
};