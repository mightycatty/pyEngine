
import tensorflow as tf


class InferenceWithTFLite:
    def __init__(self, model_file,
                 pre_processing_fn=None,
                 post_processing_fn=None,
                 **kwargs):
        self.pb_dir = model_file
        self.pre_processing_fn = pre_processing_fn  # lambda x: fn(x, **kwargs)
        self.post_processing_fn = post_processing_fn
        self._init_interpreter()

    def _init_interpreter(self):
        self.interpreter = tf.lite.Interpreter(model_path=self.pb_dir)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.input_shape = self.input_details[0]['shape']
        self.output_details = self.interpreter.get_output_details()
        self.output_shape = self.output_details[0]['shape']

    def predict(self, input_data, **kwargs):
        if self.pre_processing_fn:
            input_data = self.pre_processing_fn(input_data)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output_details[0]['index'])
        if self.post_processing_fn:
            result = self.post_processing_fn(result)
        return result
