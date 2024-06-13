# LegacyCNN

Libraries to integrate CNN models into legacy C++ systems without using Torchscript

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
