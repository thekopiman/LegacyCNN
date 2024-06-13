# LegacyCNN

Libraries to integrate CNN models into legacy C++ systems without using Torchscript

# Important: Conv1d discrepancy

The `Conv1d` layer used by Speechbrain is different from the one from Pytorch.

Speechbrain conv1d [docs](https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/nnet/CNN.html#Conv1d)

Likewise, the other layers utilised by speechbrain models (eg. `Conv2d`) are different as well. However, this repo will not replicate them. Only `Conv1d` will be replicated.

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
