# LegacyCNN

Libraries to integrate CNN models into legacy C++ systems without using Torchscript.

Also, this library requires the memory of the various layers/models to be **_allocated during compilation_**. A key factor in many embedded systems as dynamic allocation during execution (eg. malloc, calloc, etc) might lead to errors.

# Important: Conv1d discrepancy

The `Conv1d` layer used by Speechbrain is different from the one from Pytorch.

Speechbrain conv1d [docs](https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/nnet/CNN.html#Conv1d)

If you set the \*args of `Conv1d` (Speechbrain) to default; without changing any padding arguments (padding, padding_mode, default_padding).

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

Then, a `F.pad(x, (p/2, p/2), "reflect)` will be applied to the input before convolution. The value of p is selected such that the output dimension is equal to the input dimension.

Likewise, the other layers utilised by speechbrain models (eg. `Conv2d`) are different as well. However, this repo will not replicate them. Only `Conv1d` will be replicated in C++.

## Conv1d (SpeechBrain)

Default Arguments:

```
out_channels,
kernel_size,
input_shape=None,
in_channels=None,
stride=1,
dilation=1,
padding="same",
groups=1,
bias=True,
padding_mode="reflect",
skip_transpose=False,
weight_norm=False,
conv_init=None,
default_padding=0,
```

# Pytorch: Model creation

Create a model in PyTorch. \
Add another method called `save` within the model to allow for the weights to be saved in `.bin`.

```
def save(self, dirpath=""):
        layer0_conv = SaveAsBin(self.layer0[0], "layer0_conv", dirpath)
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

```bash
g++ -O2 *.cpp layers/*.h models/*.h utils/*.h -o test
test
```
