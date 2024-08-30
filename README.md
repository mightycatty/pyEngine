# pyEngine

A wrapper engine for easily experimenting with different model inference engines.

## Supported Inference Engines/Frameworks

- **TensorFlow** (.pb)
- [TensorFlow Lite](https://www.tensorflow.org/lite) (.tflite)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) (.onnx)
- [OpenVINO](https://software.intel.com/en-us/openvino-toolkit) (.xml)
- [TensorRT](https://developer.nvidia.com/tensorrt) (.trt)
- [MNN](https://github.com/alibaba/MNN) (.mnn)

## Installation

```bash
git clone https://github.com/mightycatty/pyEngine
cd pyEngine && pip install -e .
```

## Usage 
```python
from pyEngine import IE

model_dir = 'example.onnx'  # Your converted IR model path
model = IE(model_dir, *args, **kwargs)

input_data = None  # Your input data here
result = model.predict(input_data)

```