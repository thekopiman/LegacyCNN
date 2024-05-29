#ifndef conv1d_h
#define conv1d_h

// out_dim = (input_width + 2*pad - dilation*(kernal - 1) - 1)/stride + 1
template <int kernal, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_dim>
class Conv1d
{
public:
	Conv1d()
	{
		// Initialize kernal weights to 0
		// The matrix has been automatically dilated
		for (int i = 0; i < channel_out; ++i)
		{
			for (int j = 0; j < channel_in; ++j)
			{
				for (int k = 0; k < (kernal - 1) * dilation + 1; ++k)
				{
					this->matrix[i][j][k] = 1.0f; // Initialize with 0.0f
				}
			}
		}

		// Initialize bias to 0
		for (int i = 0; i < channel_out; ++i)
		{
			this->bias[i] = 0.0f; // Initialize with 0.0f
		}

		// Initialize empty_input to 0
		for (int i = 0; i < channel_in; ++i)
		{
			for (int j = 0; j < input_width; ++j)
			{

				this->empty_input[i][j] = 0.0f; // Initialize with 0.0f
			}
		}
	};
	void setWeights(float (&new_weights)[channel_out][channel_in][kernal])
	{
		// New weights are being set
		for (int i = 0; i < channel_out; i++)
		{
			for (int j = 0; j < channel_in; j++)
			{
				for (int k = 0; k < kernal; k++)
				{
					this->matrix[i][j][k * dilation] = new_weights[i][j][k];
				}
			}
		}
	};
	void setBias(float (&new_bias)[channel_out])
	{
		// New bias are being set
		for (int i = 0; i < channel_out; i++)
		{
			this->bias[i] = new_bias[i];
		}
	};
	void getOutput(float (&input)[channel_in][input_width], float (&output)[channel_out][out_dim])
	{
		// Here we assume that the dim of input/output is correct -> else it will lead to errors

		// Artificially pad the input values using empty_input
		for (int i = 0; i < channel_in; i++)
		{
			for (int j = 0; j < input_width; j++)
			{
				this->empty_input[i][j + pad] = input[i][j];
			}
		}

		// Computation of the values
		int total;
		for (int i = 0; i < out_dim; i++)
		{
			for (int j = 0; j < channel_out; j++)
			{
				total = this->bias[j];

				// Individual computation
				for (int b = 0; b < channel_in; b++)
				{
					for (int k = 0; k < (kernal - 1) * dilation + 1; k++)
					{
						total += matrix[j][b][k] * empty_input[b][i * stride + k];
					}
				}

				output[j][i] = total;
			}
		}
	};

private:
	float matrix[channel_out][channel_in][(kernal - 1) * dilation + 1];
	float bias[channel_out];
	int empty_input[channel_in][input_width + 2 * pad];
};

#endif