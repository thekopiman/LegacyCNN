# LegacyCNN

Libraries to integrate CNN models into legacy C++ systems without using Torchscript

# Important: Conv1d discrepancy

The `Conv1d` layer used by Speechbrain is different from the one from Pytorch.

Speechbrain conv1d [docs](https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/nnet/CNN.html#Conv1d)

If you set the \*args of `Conv1d` (Speechbrain) to default; without changing any padding arguments (padding, padding_mode, default_padding).\

Example of basic `Conv1d` usage

```
layer = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
        )
```

Then, a basic `F.pad(x, (p, 0))` (Left pad) will be applied to the input before `matmul`. The value of p is selected such that the output dimension is equal to the input dimension.

Likewise, the other layers utilised by speechbrain models (eg. `Conv2d`) are different as well. However, this repo will not replicate them. Only `Conv1d` will be replicated in C++.

# Pytorch: Model creation

Create a model in PyTorch. \
Add another method called `save` within the model to allow for the weights to be saved in `.bin`.

```
def save(self, dirpath=""):
        layer0_conv = SaveAsByte(self.layer0[0], "layer0_conv", dirpath)
        ...

        layer0_conv.saveBoth()
        ...

        layer0_bn.saveBoth()
        ...
```

Note: For layers (Sequential blocks) that consist of individual sub-layers, you have to extract them out here.

# C++:

Make sure you load the weights via the `.bin` files before running `forward`.

### Compile & Run as follows

```
g++ -O2 *.cpp layers/*.h models/*.h utils/*.h -o test
test
```
