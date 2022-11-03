from .base_ie import BaseIE
import onnxruntime as rt
import numpy as np

class InferenceWithOnnx(BaseIE):
    def _init_session(self, model_file, *args, **kwargs):
        self.sess = rt.InferenceSession(model_file)
        self.input_names = [item.name for item in self.sess.get_inputs()]
        self.input_shapes = [item.shape for item in self.sess.get_inputs()]
        if (self.input_shapes[0][-1] != 3):  # detect nhwc/nchw automatically
            self._nchw = True
        self.output_names = [item.name for item in self.sess.get_outputs()]
        self.output_shapes = [item.shape for item in self.sess.get_outputs()]

    def _predict(self, input_data,
                *args, **kwargs):
        feed_dict = {}
        if(np.ndim(input_data) != len(self.input_shapes[0])):
            input_data = np.expand_dims(input_data, axis=0)
        input_data = [input_data] # TODO: to support multi-inputs
        for key, value in zip(self.input_names, input_data):
            feed_dict[key] = value
        result = self.sess.run(self.output_names, input_feed=feed_dict)
        result = result[0] if len(self.output_names) == 1 else result
        return result
