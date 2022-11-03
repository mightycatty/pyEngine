import tensorflow as tf
import tensorflow.python.keras.backend as K
import numpy as np


def _wrap_model(model_fn, weight_path, interested_node_list=None, input_shape=(None, None, 3)):
    """
    封装模型，以感兴趣节点作为输出, 不给顶opt点时，直接输出模型输出
    :param model_fn:
    :param weight_path:
    :param interested_node_list:
    :return:
    """
    tf.keras.backend.set_image_data_format('channels_last')
    input_tensor = tf.keras.layers.Input(input_shape, name='input')
    segmentation_output = model_fn(input_tensor)
    model = tf.keras.models.Model(input_tensor, segmentation_output)
    model_dir = weight_path
    model.load_weights(model_dir, by_name=True)
    if interested_node_list is None:
        layers_name = model.output.name
        inter_model = model
    else:
        layers_name = interested_node_list # layers_name = ['concatenate', 'multiply']
        interested_tensors = [model.get_layer(item).output for item in layers_name]
        inter_model = tf.keras.models.Model(model.input, interested_tensors)
    inter_model.trainable = False
    return inter_model, layers_name


def _load_model(model_fn, weight_path, interested_node_list=None, input_shape=(None, None, 3)):
    model, output_names = _wrap_model(model_fn, weight_path, interested_node_list=interested_node_list, input_shape=input_shape)
    return model, output_names


class InferenceKerasModel(object):
    """
    封装tensorflow keras model，用于inference
    """
    def __init__(self, keras_model=None, model_fn=None, model_weight_dir=None, input_shape=(None, None, 3), 
                 pre_processing_fn=None, post_processing_fn=None):
        self.pre_processing_fn = pre_processing_fn
        self.post_processing_fn = post_processing_fn
        self._session_config()
        with self.sess.as_default():
            if keras_model is None:
                assert model_fn and model_weight_dir, 'model fn and model weight must be provided when keras model is None'
                keras_model, _ = _load_model(model_fn, model_weight_dir, input_shape=input_shape)
            self.unit_model = keras_model
            self._inference_mode_setting()

    def _inference_mode_setting(self):
        K.set_learning_phase(0)
        for item in self.unit_model.layers:
            item.trainable = False
        self.unit_model.trainable = False

    def _session_config(self):
        K.clear_session()
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.01
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        K.set_session(self.sess)

    def predict(self, input_data):
        if self.pre_processing_fn:
            input_data = self.pre_processing_fn(input_data)
        result = self.unit_model.predict(input_data)
        if self.post_processing_fn:
            result = self.post_processing_fn(result)
        return result