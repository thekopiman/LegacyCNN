/**
 * @file conv1d.h
 * @author Kok Chin Yi (kchinyi@dso.org.sg)
 * @brief
 * @version 0.1
 * @date 2024-06-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef conv1d_h
#define conv1d_h

#include <fstream>
#include <assert.h>
#include <string>
#include <iostream>

/**
 * @brief Convolution 1D
 *
 * https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
 *
 * Here out_width = (input_width + pad - dilation*(kernel - 1) - 1)/stride + 1
 *
 * Padding mode:
 * 0 - Normal padding (at both sides)
 * 1 - Left Pad
 * 2 - Right Pad
 * 3 - Reflect Pad
 *
 * The "pad" here actually refers to 2*pad. So the final input_width after padding is input_width + pad.
 *
 * For the case of Normal padding and Reflect pad, pad/2 will be applied to both sides.
 *
 * @tparam kernel
 * @tparam stride
 * @tparam channel_in
 * @tparam channel_out
 * @tparam pad
 * @tparam dilation
 * @tparam input_width
 * @tparam out_width
 * @tparam T
 */
template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_width, typename T>
class Conv1d
{
public:
	/**
	 * @brief Construct a new Conv1d<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_width, T>::Conv1d object
	 *
	 * Set default padding mode of Normal Padding
	 *
	 * @tparam kernel
	 * @tparam stride
	 * @tparam channel_in
	 * @tparam channel_out
	 * @tparam pad
	 * @tparam dilation
	 * @tparam input_width
	 * @tparam out_width
	 * @tparam T
	 */
	Conv1d();

	/**
	 * @brief Construct a new Conv1d<kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_width, T>::Conv1d object
	 *
	 * Padding mode:
	 * 0 - Normal padding (at both sides)
	 * 1 - Left Pad
	 * 2 - Right Pad
	 * 3 - Reflect Pad
	 *
	 * @tparam kernel
	 * @tparam stride
	 * @tparam channel_in
	 * @tparam channel_out
	 * @tparam pad
	 * @tparam dilation
	 * @tparam input_width
	 * @tparam out_width
	 * @tparam T
	 * @param mode
	 */
	Conv1d(int padding_mode);

	void setWeights(T (&new_weights)[channel_out][channel_in][kernel]);
	void setBias(T (&new_bias)[channel_out]);

	// Overloaded - Flat weights
	void setWeights(T (&new_weights)[channel_out * channel_in * kernel]);

	// Overloading from pathname
	void setWeights(std::string pathname, bool displayWeights);
	void setBias(std::string pathname, bool displayBias);

	// Overloading from infile
	void setWeights(std::ifstream &infile, bool displayWeights);
	void setBias(std::ifstream &infile, bool displayBias);

	/**
	 * @brief Load the weights via pathname
	 *
	 * @tparam kernel
	 * @tparam stride
	 * @tparam channel_in
	 * @tparam channel_out
	 * @tparam pad
	 * @tparam dilation
	 * @tparam input_width
	 * @tparam out_width
	 * @tparam T
	 * @param pathname
	 */
	void loadweights(std::string pathname);

	/**
	 * @brief Load the weights via infile
	 *
	 * @tparam kernel
	 * @tparam stride
	 * @tparam channel_in
	 * @tparam channel_out
	 * @tparam pad
	 * @tparam dilation
	 * @tparam input_width
	 * @tparam out_width
	 * @tparam T
	 * @param infile
	 */
	void loadweights(std::ifstream &infile);

	/**
	 * @brief Performs forward feed
	 *
	 * @tparam kernel
	 * @tparam stride
	 * @tparam channel_in
	 * @tparam channel_out
	 * @tparam pad
	 * @tparam dilation
	 * @tparam input_width
	 * @tparam out_width
	 * @tparam T
	 * @param input
	 * @param output
	 */
	void forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_width]);

	// For debugging purposes
	void printempty_input();

	~Conv1d();

private:
	T matrix[channel_out][channel_in][kernel];
	T flat_matrix[channel_out * channel_in * kernel];
	T bias[channel_out];
	T empty_input[channel_in][input_width + pad];
	int padding_mode = 0;

	/**
	 * @brief Initialization to set the parameters to default values.
	 *
	 */
	void init();
	/**
	 * @brief Pads the input in accordance to the padding mode and pad specified.
	 *
	 * @tparam kernel
	 * @tparam stride
	 * @tparam channel_in
	 * @tparam channel_out
	 * @tparam pad
	 * @tparam dilation
	 * @tparam input_width
	 * @tparam out_width
	 * @tparam T
	 * @param input
	 */
	void padInput(T (&input)[channel_in][input_width]);
};

#include "conv1d.cpp"

#endif