#ifndef conv1d_h
#define conv1d_h

#include <fstream>
#include <assert.h>
#include <string>
#include <iostream>

// out_dim = (input_width + 2*pad - dilation*(kernel - 1) - 1)/stride + 1
template <int kernel, int stride, int channel_in, int channel_out, int pad, int dilation, int input_width, int out_dim>
class Conv1d
{
public:
	Conv1d()
	{
		// Initialize kernel weights to 0
		// The matrix has been automatically dilated
		for (int i = 0; i < channel_out; ++i)
		{
			for (int j = 0; j < channel_in; ++j)
			{
				for (int k = 0; k < kernel; ++k)
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
	void setWeights(float (&new_weights)[channel_out][channel_in][kernel])
	{
		// New weights are being set
		for (int i = 0; i < channel_out; i++)
		{
			for (int j = 0; j < channel_in; j++)
			{
				for (int k = 0; k < kernel; k++)
				{
					this->matrix[i][j][k] = new_weights[i][j][k];
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

	// Overloading from filename
	void setWeights(std::string filename, bool displayWeights)
	{
		std::ifstream infile(filename, std::ios::binary);
		if (!infile)
		{
			std::cout << "Error opening file!" << std::endl;
			return;
		}
		// Read dimensions
		int dim1, dim2, dim3;
		infile.read(reinterpret_cast<char *>(&dim1), sizeof(int));
		infile.read(reinterpret_cast<char *>(&dim2), sizeof(int));
		infile.read(reinterpret_cast<char *>(&dim3), sizeof(int));

		// Sanity Check
		assert(dim1 == channel_out);
		assert(dim2 == channel_in);
		assert(dim3 == kernel);

		// Calculate total size
		int total_size = dim1 * dim2 * dim3;

		// Read flattened array
		infile.read(reinterpret_cast<char *>(this->flat_matrix), total_size * sizeof(float));
		infile.close();

		for (int i = 0; i < dim1; ++i)
		{
			for (int j = 0; j < dim2; ++j)
			{
				for (int k = 0; k < dim3; ++k)
				{
					this->matrix[i][j][k] = this->flat_matrix[i * dim2 * dim3 + j * dim3 + k];
				}
			}
		}

		// Output to verify correctness
		if (displayWeights)
		{
			std::cout << "Read 3D array dimensions: (" << dim1 << ", " << dim2 << ", " << dim3 << ")\n";
			for (int i = 0; i < dim1; ++i)
			{
				for (int j = 0; j < dim2; ++j)
				{
					for (int k = 0; k < dim3; ++k)
					{
						std::cout << this->matrix[i][j][k] << " ";
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
			}
		}
	}

	// Overloading from filename
	void setBias(std::string filename, bool displayBias)
	{
		std::ifstream infile(filename, std::ios::binary);
		if (!infile)
		{
			std::cout << "Error opening file!" << std::endl;
			return;
		}
		// Read dimensions
		int dim1;
		infile.read(reinterpret_cast<char *>(&dim1), sizeof(int));

		// Sanity Check
		assert(dim1 == channel_out);

		// Calculate total size
		int total_size = dim1;

		// Read flattened array
		infile.read(reinterpret_cast<char *>(this->bias), total_size * sizeof(float));
		infile.close();

		// Output to verify correctness
		if (displayBias)
		{
			std::cout << "Read 3D array dimensions: (" << dim1 << ")\n";
			for (int i = 0; i < dim1; ++i)
			{
				std::cout << this->bias[i] << " ";
			}
			std::cout << std::endl;
		}
	}
	void padInput(float (&input)[channel_in][input_width])
	{
		for (int i = 0; i < channel_in; i++)
		{
			for (int j = 0; j < input_width; j++)
			{
				this->empty_input[i][j + pad] = input[i][j];
			}
		}
	}
	void getOutput(float (&input)[channel_in][input_width], float (&output)[channel_out][out_dim])
	{
		// Here we assume that the dim of input/output is correct -> else it will lead to errors

		// Artificially pad the input values using empty_input
		padInput(input);

		// Computation of the values
		for (int j = 0; j < channel_out; j++)
		{
			for (int i = 0; i < out_dim; i++)
			{
				output[j][i] = this->bias[j];

				// Individual computation
				for (int b = 0; b < channel_in; b++)
				{
					for (int k = 0; k < kernel; k++)
					{
						output[j][i] += this->matrix[j][b][k] * this->empty_input[b][i * stride + k * dilation];
					}
				}
			}
		}
	};

private:
	float matrix[channel_out][channel_in][kernel];
	float flat_matrix[channel_out * channel_in * kernel];
	float bias[channel_out];
	float empty_input[channel_in][input_width + 2 * pad];
};

#endif