# @Time    : 2020/12/28 17:56
# @Author  : Heshuai
# @Email   : heshuai.sec@gmail.com
import MNN
from .base_ie import BaseIE
import numpy as np


# TODO: output seems abnormal
class InferenceWithMNN(BaseIE):
    def _init_session(self, model_file, *args, **kwargs):
        self._interpreter = MNN.Interpreter(model_file)
        self._session = self._interpreter.createSession()
        self._input_tensor = self._interpreter.getSessionInput(self._session)
        self.input_shape = self._input_tensor.getShape()
        self._output_tensor = self._interpreter.getSessionOutput(self._session)
        self.output_shape = self._output_tensor.getShape()
        if(self.input_shape[-1] < 16): # detect nhwc/nchw automatically
            self._nchw = True

    def _predict(self, input_data, *args, **kwargs):
        if self._nchw:
            tensor_type = MNN.Tensor_DimensionType_Caffe
        else:
            tensor_type = MNN.Tensor_DimensionType_Tensorflow
        input_data = np.expand_dims(input_data, axis=0)
        tmp_input = MNN.Tensor(self.input_shape, MNN.Halide_Type_Float, \
                               input_data, tensor_type)
        self._input_tensor.copyFrom(tmp_input)
        self._interpreter.runSession(self._session)
        output_tensor = self._interpreter.getSessionOutput(self._session)
        # constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
        tmp_output = MNN.Tensor(self.output_shape, MNN.Halide_Type_Float, np.ones(self.output_shape).astype(np.float32),
                                MNN.Tensor_DimensionType_Caffe)
        output_tensor.copyToHostTensor(tmp_output)
        tmp_output = tmp_output.getData()
        tmp_output = np.array(tmp_output)
        tmp_output = tmp_output.reshape(*self.output_shape)
        return tmp_output

