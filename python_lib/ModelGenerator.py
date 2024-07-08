import math


def get_padding_value(L_in: int, stride: int, kernel_size: int, dilation: int):
    """This function computes the number of elements to add for zero-padding.

    Arguments
    ---------
    L_in : int
    stride: int
    kernel_size : int
    dilation : int

    Returns
    -------
    padding : int
        The size of the padding to be added
    """
    if stride > 1:
        padding = math.floor(kernel_size / 2)

    else:
        L_out = math.floor((L_in - dilation * (kernel_size - 1) - 1) / stride) + 1
        padding = math.floor((L_in - L_out) / 2)

    return padding


class BasicLayer:
    freq = 0  # Static Variable

    def __init__(self, channel_in: int, channel_out: int, T: type):
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.T = T.__name__
        self.input_width = None
        self.output_width = None
        self.mode = None
        self.freq = BasicLayer.freq
        BasicLayer.freq += 1

    def forward(self, x_shape):
        assert x_shape[0] == self.channel_in
        self.input_width = x_shape[1]
        self.output_width = x_shape[1]

        return x_shape

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.freq}"

    def initialisedstring(self):
        if self.input_width == None or self.output_width == None:
            raise Exception("Run .forward before generating the string")

    def return_mode(self):
        if self.mode is not None:
            return f"{self}.{self.mode}"


class Conv1d(BasicLayer):
    def __init__(
        self,
        channel_in,
        channel_out,
        kernel=3,
        stride=1,
        dilation=1,
        pad="auto",
        padding_mode="reflect",
        T=float,
    ):
        super().__init__(channel_in, channel_out, T)
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.mode = padding_mode

        assert self.pad == "auto" or type(self.pad) == int

    def forward(self, x_shape):
        assert x_shape[0] == self.channel_in

        self.input_width = x_shape[1]

        if self.pad == "auto":
            self.pad = get_padding_value(
                x_shape[1], self.stride, self.kernel, self.dilation
            )
        L_out = math.floor(
            (x_shape[1] + 2 * self.pad - self.dilation * (self.kernel - 1) - 1)
            / self.stride
            + 1
        )

        self.output_width = L_out

        return (self.channel_out, L_out)

    def initialisedstring(self):
        super().initialisedstring()

        return f"{self.__class__.__name__}<{self.kernel}, {self.stride}, {self.channel_in}, {self.channel_out}, {self.pad*2}, {self.dilation}, {self.input_width}, {self.output_width}, {self.T}> {self}; \n{self.T} x{self.freq}[{self.channel_out}][{self.output_width}];"


class TDNNBlock(Conv1d):
    def __init__(
        self, channel_in, channel_out, kernel=3, stride=1, dilation=1, T=float
    ):
        super().__init__(
            channel_in, channel_out, kernel, stride, dilation, "auto", None, T
        )

    def initialisedstring(self):
        super().initialisedstring()

        return f"{self.__class__.__name__}<{self.kernel}, {self.stride}, {self.channel_in}, {self.channel_out}, {self.dilation}, {self.input_width}, {self.output_width}, {self.pad*2}, {self.T}> {self}; \n{self.T} x{self.freq}[{self.channel_out}][{self.output_width}];"


class BatchNorm1d(BasicLayer):
    def __init__(self, channel, T=float):
        super().__init__(channel_out=channel, channel_in=channel, T=T)

    def initialisedstring(self):
        super().initialisedstring()

        return f"{self.__class__.__name__}<{self.channel_in}, {self.input_width}, {self.T}> {self}; \n{self.T} x{self.freq}[{self.channel_out}][{self.output_width}];"


class SEBlock(BasicLayer):
    def __init__(self, channel, se_channel, T=float):
        super().__init__(channel_in=channel, channel_out=channel, T=T)
        self.se_channel = se_channel

    def initialisedstring(self):
        super().initialisedstring()

        return f"{self.__class__.__name__}<{self.channel_in}, {self.se_channel}, {self.channel_out}, {self.input_width}, {self.output_width}, {self.T}> {self}; \n{self.T} x{self.freq}[{self.channel_out}][{self.output_width}];"


class Res2NetBlock(TDNNBlock):
    def __init__(self, scale, channel_in, channel_out, kernel=3, dilation=1, T=float):
        super().__init__(channel_in, channel_out, kernel, 1, dilation, T)
        self.scale = scale

    def initialisedstring(self):
        super().initialisedstring()

        return f"{self.__class__.__name__}<{self.kernel}, {self.channel_in}, {self.channel_out}, {self.dilation}, {self.input_width}, {self.output_width}, {self.pad*2},{self.scale}, {self.T}> {self}; \n{self.T} x{self.freq}[{self.channel_out}][{self.output_width}];"


class SERes2NetBlock(Res2NetBlock):
    def __init__(
        self,
        res2net_scale,
        se_channel,
        channel_in,
        channel_out,
        kernel=3,
        dilation=1,
        T=float,
    ):
        super().__init__(res2net_scale, channel_in, channel_out, kernel, dilation, T)
        self.se_channel = se_channel

    def initialisedstring(self):
        super().initialisedstring()

        return f"{self.__class__.__name__}<{self.kernel}, {self.channel_in}, {self.channel_out}, {self.dilation}, {self.input_width}, {self.output_width}, {self.pad*2}, {self.scale}, {self.se_channel}, {self.T}> {self}; \n{self.T} x{self.freq}[{self.channel_out}][{self.output_width}];"


class ASP(BasicLayer):
    def __init__(self, channels, attention_channels, T: type = float):
        super().__init__(channels, channels * 2, T)
        self.attention_channels = attention_channels

    def forward(self, x_shape):
        assert x_shape[0] == self.channel_in
        self.input_width = x_shape[1]
        self.output_width = 1

        return (self.channel_out, self.output_width)

    def initialisedstring(self):
        return f"{self.__class__.__name__}<{self.channel_in}, {self.attention_channels}, {self.input_width}, {self.output_width}, {self.T}> {self}; \n{self.T} x{self.freq}[{self.channel_out}][{self.output_width}];"


class Dense:
    freq = 0  # Static Variable

    def __init__(self, output_width, T: type = float):
        self.output_width = output_width
        self.input_width = None
        self.T = T.__name__
        self.freq = Dense.freq
        Dense.freq += 1

    def forward(self, x_shape: int):
        self.input_width = x_shape

        return self.output_width

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.freq}"

    def initialisedstring(self):
        if self.input_width == None or self.output_width == None:
            raise Exception("Run .forward before generating the string")
        else:
            return f"{self.__class__.__name__}<{self.input_width}, {self.output_width}, {self.T}> {self}; \n{self.T} y{self.freq}[{self.output_width}];"


def Cat(xl: list):
    total_channels = 0
    for x in xl:
        total_channels += x[0]

    return (total_channels, xl[0][1])


if __name__ == "__main__":
    pass
