import torch
import torch.nn.functional as F
import torch.nn as nn
from .saveasfile import SaveAsBin


# class InputTransform:
#     def __init__(self, output_size=128):
#         self.output_size = output_size

#     def __call__(self, x):
#         batch_size, n_channels, input_size = x.size()

#         x_reshaped = x

#         if input_size > self.output_size:
#             # Resize using bilinear interpolation
#             x_reshaped = F.interpolate(x, size=(self.output_size), mode='linear', align_corners=False)

#         # Reshape back to the original shape
#         x_reshaped = x_reshaped.view(batch_size, n_channels, -1)

#         # Pad if necessary
#         pad_size = self.output_size - x_reshaped.size(2)
#         left_pad = pad_size // 2
#         right_pad = pad_size - left_pad
#         x_padded = F.pad(x_reshaped.unsqueeze(3), (0, 0, left_pad, right_pad), mode='constant', value=0)
#         x_padded = x_padded.squeeze(3)

#         return x_padded


def BasicBlock(in_channels, out_channels, kernel, stride=1):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel, stride),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(out_channels),
    )


class BasicModel(torch.nn.Module):
    def __init__(self, input_size=2, input_length=64, out_features=6):
        super().__init__()

        self.inputs = input_size

        # self.input_layer = InputTransform(output_size = input_length)

        self.layer0 = BasicBlock(self.inputs, 4, 3)
        self.layer1 = BasicBlock(4, 4, 3)
        self.layer2 = BasicBlock(4, 4, 3)
        self.layer3 = BasicBlock(4, 4, 3)
        self.layer4 = BasicBlock(4, 4, 3)

        # Final Dense Layer
        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=4 * (input_length - 10), out_features=out_features),
        )

    def forward(self, x):
        y = self.layer0(x)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        ## Final Layer
        y = self.final(y)

        z = F.softmax(y)
        return z

    def return_layers(self):

        lst = []

        # Do not need to save ReLU as ReLU has no weights
        lst.append(self.layer0[0])  # Conv layer
        lst.append(self.layer0[2])  # BN Layer

        lst.append(self.layer1[0])
        lst.append(self.layer1[2])

        lst.append(self.layer2[0])
        lst.append(self.layer2[2])

        lst.append(self.layer3[0])
        lst.append(self.layer3[2])

        lst.append(self.layer4[0])
        lst.append(self.layer4[2])

        # Do not need to save Flatten as Flatten has no weights
        lst.append(self.final[1])  # FC weights

        return lst
