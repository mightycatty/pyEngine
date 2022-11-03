from abc import ABCMeta, abstractmethod
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np


class BaseIE:
    __metaclass__ = ABCMeta
    def __init__(self, model_file,
                    pre_processing_fn=None,
                    post_processing_fn=None,
                    input_nodes=None,
                    output_nodes=None,
                    *args, **kwargs):
        self._pre_processing_fn = pre_processing_fn
        self._post_processing_fn = post_processing_fn
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._nchw = False
        self._session_init_flag = False # set to true when self._init_ was called
        self._init_session(model_file, *args, **kwargs)
        self._session_init_flag = True
    @abstractmethod
    def _init_session(self, *args, **kwargs):
        return
    @abstractmethod
    def _predict(self, input_data, *args, **kwargs):
        return

    def get_inputs(self):
        """
        return intput nodes as dict{name:shape, name_1:shape_1_for_name_1}
        :return:
        """
        input_nodes = {}
        return input_nodes

    def get_outputs(self):
        output_nodes = {}
        return output_nodes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO: close active session
        del self

    def __del__(self):
        # TODO： close active session
        pass

    def __call__(self, *args, **kwargs):
        return self._predict(*args, **kwargs)

    def _input_shape_check(self, input_data):
        #todo:
        return input_data

    def predict(self, input_data,
                *args, **kwargs):
        # TODO: to support multi-inputs
        assert self._session_init_flag, 'no activate session!'
        if self._pre_processing_fn is not None:
            input_data = self._pre_processing_fn(input_data, *args, **kwargs)
        if self._nchw:
            if np.ndim(input_data) == 3:
                input_data = input_data.transpose((2, 0, 1))
            if np.ndim(input_data) == 4:
                input_data = input_data.transpose((0, 3, 1, 2))
        input_data = np.float32(input_data)
        result = self._predict(input_data, *args, **kwargs)
        if self._post_processing_fn is not None:
            result = self._post_processing_fn(result)
        return result