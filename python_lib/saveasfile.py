import torch
import numpy as np
import os


class SaveAsByte:
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
