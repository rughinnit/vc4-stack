"""Name-only transforms for the ONNX export path (never applied)."""


class _NotUsed:
    def __init__(self, *a, **k):
        raise NotImplementedError("torchvision shim: transforms are not used by ONNX export")


Compose = ToTensor = Normalize = Resize = CenterCrop = PILToTensor = _NotUsed
