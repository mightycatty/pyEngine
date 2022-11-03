"""universal inference engine for valid IRs
Usage:
    model_dir = None #your mnn/onnx/tflite model
    ie = IE(model_dir)
    input_data = None
    ie.predict(input_data)
"""
import logging

logger = logging.getLogger(__name__)

VALID_IE_DICT ={}
try:
    from .mnn_ie import InferenceWithMNN
    VALID_IE_DICT['mnn'] = InferenceWithMNN
except:
    logger.warning('mnn not valid')
try:
    from .onnx_ie import InferenceWithOnnx
    VALID_IE_DICT['onnx'] = InferenceWithOnnx
except:
    logger.warning('onnx not valid')
try:
    from .tflite_ie import InferenceWithTFLite
    VALID_IE_DICT['tflite'] = InferenceWithTFLite
except:
    logger.warning('tflite not valid')

try:
    from .tensorflow_ie import InferenceWithPb
    VALID_IE_DICT['pb'] = InferenceWithPb
except:
    logger.warning('tensorflow not valid')

__all__ = ['IE']


class IE:
    def __init__(self, model_dir, *args, **kwargs):
        model_format = model_dir.split('.')[-1]
        if model_format not in VALID_IE_DICT.keys():
            logger.error('package for {} not installed!'.format(model_format))
            exit(0)
        else:
            self.ie = VALID_IE_DICT[model_format](model_format, *args, **kwargs)

    def predict(self, input_data,
                *args, **kwargs):
        result = self.ie.predict(input_data, *args, **kwargs)
        return result

