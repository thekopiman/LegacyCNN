import torch
import numpy as np
import os


class SaveAsBin:
    def __init__(self, layer, layername, dirpath):
        self.layer = layer
        self.layername = layername
        self.dirpath = dirpath

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    def saveBias(self, transpose=False):
        bias = self.layer.bias.cpu().detach().numpy()
        if transpose:
            bias = np.transpose(bias)
        dim = bias.shape

        flatten_bias = bias.flatten()

        with open(os.path.join(self.dirpath, f"{self.layername}_bias.bin"), "wb") as f:
            # Write the dimensions down
            f.write(np.array(dim, dtype=np.int32).tobytes())
            # Write the flatten bias down
            f.write(flatten_bias.tobytes())

    def saveWeights(self, transpose=False):
        weights = self.layer.weight.cpu().detach().numpy()
        if transpose:
            weights = np.transpose(weights)
        dim = weights.shape

        flatten_weights = weights.flatten()

        with open(
            os.path.join(self.dirpath, f"{self.layername}_weights.bin"), "wb"
        ) as f:
            # Write the dimensions down
            f.write(np.array(dim, dtype=np.int32).tobytes())
            # Write the flatten weights down
            f.write(flatten_weights.tobytes())

    def saveBoth(self):
        self.saveBias()
        self.saveWeights()


class BlockSave:
    def __init__(self, layers: list, blockname: str, dirpath):
        self.layers = layers
        self.blockname = blockname
        self.dirpath = dirpath

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        self.save()

    def save(self):
        """
        We will save in this format
        layers[0] dimension - weights
        layers[0] weights
        layers[0] dimension - bias
        layers[0] bias
        layers[1] dimension - weights
        layers[1] weights
        layers[1] dimension - bias
        layers[1] bias
        etc
        ...

        """
        with open(os.path.join(self.dirpath, f"{self.blockname}.bin"), "wb") as f:
            for layer in self.layers:
                # Weights
                weights = layer.weight.cpu().detach().numpy()
                weight_dim = weights.shape
                flatten_weights = weights.flatten()

                # Write to bin file (weights)
                f.write(np.array(weight_dim, dtype=np.int32).tobytes())
                f.write(flatten_weights.tobytes())

                # Bias
                bias = layer.bias.cpu().detach().numpy()
                bias_dim = bias.shape
                flatten_bias = bias.flatten()

                # Write to bin file (bias)
                f.write(np.array(bias_dim, dtype=np.int32).tobytes())
                f.write(flatten_bias.tobytes())

                # For Batch Norm Only
                try:
                    # Running mean
                    running_mean = layer.running_mean.cpu().detach().numpy()
                    running_mean_dim = running_mean.shape
                    flatten_running_mean = running_mean.flatten()

                    # Write to bin file (running mean)
                    f.write(np.array(running_mean_dim, dtype=np.int32).tobytes())
                    f.write(flatten_running_mean.tobytes())

                    # Running var
                    running_var = layer.running_var.cpu().detach().numpy()
                    running_var_dim = running_var.shape
                    flatten_running_var = running_var.flatten()

                    # Write to bin file (running var)
                    f.write(np.array(running_var_dim, dtype=np.int32).tobytes())
                    f.write(flatten_running_var.tobytes())
                except AttributeError:
                    pass
