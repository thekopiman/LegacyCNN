import torch
import torch.nn.functional as F
import torch.nn as nn
from .saveasfile import SaveAsByte


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


def BasicCNNBlock(in_channels, out_channels, kernel, stride=1):
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

        self.layer0 = BasicCNNBlock(self.inputs, 4, 3)
        self.layer1 = BasicCNNBlock(4, 4, 3)
        self.layer2 = BasicCNNBlock(4, 4, 3)
        self.layer3 = BasicCNNBlock(4, 4, 3)
        self.layer4 = BasicCNNBlock(4, 4, 3)

        # Final Dense Layer
        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=4 * (input_length - 10), out_features=out_features),
        )

    def forward(self, x):
        # x = self.input_layer(x)
        y = self.layer0(x)
        print(y)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)

        ## Final Layer
        y = self.final(y)

        z = F.softmax(y)
        return z

    def save(self, dirpath=""):
        layer0_conv = SaveAsByte(self.layer0[0], "layer0_conv", dirpath)
        layer0_bn = SaveAsByte(self.layer0[2], "layer0_bn", dirpath)
        layer1_conv = SaveAsByte(self.layer1[0], "layer1_conv", dirpath)
        layer1_bn = SaveAsByte(self.layer1[2], "layer1_bn", dirpath)
        layer2_conv = SaveAsByte(self.layer2[0], "layer2_conv", dirpath)
        layer2_bn = SaveAsByte(self.layer2[2], "layer2_bn", dirpath)
        layer3_conv = SaveAsByte(self.layer3[0], "layer3_conv", dirpath)
        layer3_bn = SaveAsByte(self.layer3[2], "layer3_bn", dirpath)
        layer4_conv = SaveAsByte(self.layer4[0], "layer4_conv", dirpath)
        layer4_bn = SaveAsByte(self.layer4[2], "layer4_bn", dirpath)
        final_fc = SaveAsByte(self.final[1], "final_fc", dirpath)

        layer0_conv.saveBoth()
        layer1_conv.saveBoth()
        layer2_conv.saveBoth()
        layer3_conv.saveBoth()
        layer4_conv.saveBoth()

        layer0_bn.saveBoth()
        layer1_bn.saveBoth()
        layer2_bn.saveBoth()
        layer3_bn.saveBoth()
        layer4_bn.saveBoth()

        final_fc.saveWeights()
        final_fc.saveBias()
