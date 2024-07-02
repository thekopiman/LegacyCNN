# LegacyCNN

Libraries to integrate CNN models into legacy C++ systems without using Torchscript.

Also, this library requires the memory of the various layers/models to be **_allocated during compilation_**. A key factor in many embedded systems as dynamic allocation during execution (eg. malloc, calloc, etc) might lead to errors.

## UML Class Diagram

![umldiagiagram](imgs/UML%20Diagram.png)

Attributes are omitted to prevent clutter

## Docs

TBC

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

Then, a `F.pad(x, (p/2, p/2), "reflect")` will be applied to the input before convolution. The value of p is selected such that the output dimension is equal to the input dimension.

Likewise, the other layers utilised by speechbrain models (eg. `Conv2d`) are different as well. However, this repo will not replicate them. Only `Conv1d` will be replicated in C++.

# Pytorch: Model creation

Create a model in PyTorch. \
Add another method called `return_layers` within the model.
Only return the layers with weights.

For example, notice that the basicblock layer consist of 3 layers:

- Conv1d
- ReLU
- BatchNorm1d

However, only Conv1d and BatchNorm1d layers have weights. Hence we have to do the following:

```
def BasicBlock(in_channels, out_channels, kernel, stride=1):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel, stride),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(out_channels),
    )

class MyModel(self):
        def __init__(self):
                self.layer0 = BasicBlock(2,2,3,1)
                self.layer1 = BasicBlock(2,2,3,1)

        def return_layers():
                # We ignore ReLU layer here
                return [layer0[0], layer0[2], layer1[0], layer1[2]]
```

Note: For layers (Sequential blocks) that consist of individual sub-layers, you have to extract them out here.

### Utilise the `BlockSave` Class within python_lib/saveasfile

```
model = MyModel()
BlockSave(model.return_layers(), "mymodel", "mymodelweights")
```

Here, a `mymodelweights/mymodel.bin` file will be created.

# C++:

Make sure you load the weights via the `.bin` files before running `forward`.

The order in which you save the weights aforementioned is important. From the previous example, we see that layer0 is saved before layer1. As such, during the `loadweight` method, we have to specify as such.

```
// mymodel.h/.cpp

class MyModel(){
        public:
                void loadweights(std::string pathname){
                        std::ifstream infile(pathname, std::ios::binary);
                        if (!infile)
                        {
                                std::cout << "Error opening file!" << std::endl;
                                return;
                        }

                        Block0.loadweights(infile);
                        Block1.loadweights(infile);
                }

        private:
                BasicBlock<3, 1, 2, 4, 0, 1, 16, 14, float> Block0;
                BasicBlock<3, 1, 4, 4, 0, 1, 14, 12, float> Block1;}

```

# Examples

I have provided 2 examples in the `models` directory. The accompanying python files can be seen in

- python_lib
  - BasicModel.py `basiccnn.h`
  - modules.py `ecapa.h`
- generate_basicmodelweights.ipynb `basiccnn.h`
- generate_ecapaweights.ipynb `ecapa.h`

### Compile & Run as follows

```bash
g++ -O2 *.cpp layers/*.h models/*.h utils/*.h tests/*.h -o test
test
```
