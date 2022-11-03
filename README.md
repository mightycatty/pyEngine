# pyEngine

Wrapped python inference engine for fast model inference.

## Supported inference engines/frameworks:

- Tensorflow(.pb)
- [Tensorflow Lite](https://www.tensorflow.org/lite) (.tflite)
- [Onnx Runtime](https://github.com/microsoft/onnxruntime) (.onnx)
- [Openvino](https://software.intel.com/en-us/openvino-toolkit) (.xml)
- [Tensorrt](https://developer.nvidia.com/tensorrt) (.trt)
- [MNN](https://github.com/alibaba/MNN) (.mnn)

## Installation
`
git clone https://github.com/mightycatty/pyEngine
`

`cd pyEngine && pip install -e .`

## Usage

1. convert your model to specific IR(.mm/.pb/.tflite/.onnx/.trt)
2. Just try out your desired Inference engine

```
from pyEngine import IE
model_dir = 'example.onnx' # your converted IR
model = IE(model_dir, *args, **kwars)
input_data = None
result = model.predict(None)
```



