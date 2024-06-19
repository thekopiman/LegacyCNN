#ifndef conv1d_h
#define conv1d_h

#include <fstream>
#include <assert.h>
#include <string>
#include <iostream>

// kernel, stride, channel_in, channel_out, pad, dilation, input_width, out_dim, T
//
// out_dim = (input_width + 2*pad - dilation*(kernel - 1) - 1)/stride + 1
// Padding mode
// 0 : Normal padding (at both sides), 1 : Left Pad, 2: Right Pad
// For Normal padding make sure you allocate x2 since padding will be (p/2, p/2)
template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_dim, typename T>
class Conv1d
{
public:
	Conv1d();
	// Overloaded constructor
	Conv1d(int padding_mode);
	void setWeights(T (&new_weights)[channel_out][channel_in][kernel]);
	void setBias(T (&new_bias)[channel_out]);

	// Overloaded - Flat weights
	void setWeights(T (&new_weights)[channel_out * channel_in * kernel]);

	// Overloading from filename
	void setWeights(std::string filename, bool displayWeights);
	void setBias(std::string filename, bool displayBias);

	// Overloading from infile
	void setWeights(std::ifstream &infile, bool displayWeights);
	void setBias(std::ifstream &infile, bool displayBias);

	void forward(T (&input)[channel_in][input_width], T (&output)[channel_out][out_dim]);

	~Conv1d();

private:
	T matrix[channel_out][channel_in][kernel];
	T flat_matrix[channel_out * channel_in * kernel];
	T bias[channel_out];
	T empty_input[channel_in][input_width + pad];
	int padding_mode = 0;

	void init();
	void padInput(T (&input)[channel_in][input_width]);
};

#include "conv1d.cpp"

#endif