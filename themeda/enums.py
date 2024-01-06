from enum import Enum


class TemporalProcessorType(Enum):
    LSTM = "LSTM"
    GRU = "GRU"
    TRANSFORMER = "TRANSFORMER"
    NONE = "NONE"

    def __str__(self):
        return self.value


class DecoderType(Enum):
    UNET = "UNET"
    DIFFUSION = "DIFFUSION"
    NONE = "NONE"

    def __str__(self):
        return self.value


class ResNet(Enum):
    resnet18 = "resnet18"
    resnet34 = "resnet34"
    resnet50 = "resnet50"
    resnet101 = "resnet101"
    resnet152 = "resnet152"

    def __str__(self):
        return self.value
