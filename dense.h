#ifndef dense_h
#define dense_h

template <int input_dim, int output_dim>
class Dense
{
public:
    Dense()
    {
        // Initialize the weights with 0
        for (int i = 0; i < output_dim; i++)
        {
            for (int j = 0; j < input_dim; j++)
            {
                weights[i][j] = 0.0f;
            }
        }

        // Initialize the bias with 0
        for (int i = 0; i < output_dim; i++)
        {
            bias[i] = 0.0f;
        }
    }

    void setWeights(float (&new_weights)[input_dim][output_dim])
    {
        // New weights are being set
        for (int i = 0; i < output_dim; i++)
        {
            for (int j = 0; j < input_dim; j++)
            {
                this->weights[i][j] = new_weights[i][j];
            }
        }
    };

    void setBias(float (&new_bias)[output_dim])
    {
        // New bias are being set
        for (int i = 0; i < output_dim; i++)
        {
            this->bias[i] = new_bias[i];
        }
    };

    void getOutput(float (&input)[input_dim], float (&output)[output_dim])
    {
        for (int i = 0; i < output_dim; i++)
        {
            output[i] = this->bias[i];
            for (int j = 0; j < input_dim; j++)
            {
                output[i] += this->weights[i][j] * input[j];
            }
        }
    }

private:
    float weights[output_dim][input_dim];
    float bias[output_dim];
};

#endif