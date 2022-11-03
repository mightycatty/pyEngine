import tensorflow as tf
from tensorflow import gfile
import os


class InferenceWithPb:
    """
    compact tensorflow inference backend which takes in a froze .pb and names of input and output tensor.
    1. data feed to the network is required to be pre-processed beforehand, raw output from network is delivered without any post-processing.
        Or you can register your own preprocessing function and postprocessing function to make a abstract model unit.
    2. each Inference create a graph and a session for its own, global default graph and session kept untouched.
    3. inputs and outputs is detected automatically if not given, however this is not 100% safe, double check if any unexpectation.
    """

    def __init__(self, model_file,
                 input_nodes=None,
                 output_nodes=None,
                 pre_processing_fn=None,
                 post_processing_fn=None,
                 tf_trt=False,
                 **kwargs):
        # attr
        assert os.path.exists(model_file), 'model_file not exist!'
        self.input_name = input_nodes  # list
        self.output_name = output_nodes  # list
        self.input = []  # input tensor
        self.output = []

        self.tf_trt = tf_trt
        self.pb_dir = model_file
        self.pre_processing_fn = pre_processing_fn  # lambda x: fn(x, **kwargs)
        self.post_processing_fn = post_processing_fn
        self._construct_graph()
        self._init_session()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.sess.close()
        finally:
            pass

    def __del__(self):
        try:
            self.sess.close()
        finally:
            pass

    @staticmethod
    def _read_pb(pb_dir):
        with gfile.FastGFile(pb_dir, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def

    @staticmethod
    def _automatic_inputs_outputs_detect(graph_def):
        """
        automatically detect inputs(nodes with op='Placeholder') and outputs(nodes without output edges) given a graph_def.
        Place note that this is not 100% safe, might yield wrong inputs outputs detection, double check before carrying on
        :param graph_def:
        :return: inputs(list), outputs(list)
        """
        inputs = []
        outputs = []
        node_inputs = []
        # inputs detection
        for node in graph_def.node:
            node_inputs += node.input
            if node.op == 'Placeholder':
                inputs.append(node.name + ':0')
        # outputs detection
        node_inputs = list(set(node_inputs))
        for node in graph_def.node:
            if node.name not in node_inputs:
                if node.input:
                    outputs.append(node.name + ':0')
        return inputs, outputs

    # @staticmethod
    # def _trt_graph(graph_def, outputs):
    #     graph_def = trt.create_inference_graph(
    #         input_graph_def=graph_def,
    #         outputs=outputs,
    #         precision_mode='FP16',
    #         max_workspace_size_bytes = 1 << 30)
    #     return graph_def

    def _construct_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_def = InferenceWithPb._read_pb(self.pb_dir)
            au_inputs, au_outputs = InferenceWithPb._automatic_inputs_outputs_detect(graph_def)
            tf.import_graph_def(graph_def, name='')
            graph = tf.get_default_graph()
            if self.input_name is None:
                self.input_name = au_inputs
            if self.output_name is None:
                self.output_name = au_outputs
            if isinstance(self.input_name, str):
                self.input_name = [self.input_name]
            if isinstance(self.output_name, str):
                self.output_name = [self.output_name]
            self.input = [graph.get_tensor_by_name(item) for item in self.input_name]
            if self.output_name:
                self.output = [graph.get_tensor_by_name(item) for item in self.output_name]

    def _init_session(self):
        # without below configuration, raise error on tf_gpu_1.14
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(config=config, graph=self.graph)

    def predict(self, input_data,
                output_nodes=None,
                **kwargs):
        output_nodes_list = []
        if output_nodes:
            if isinstance(output_nodes, str):
                output_nodes = [output_nodes]
            assert isinstance(output_nodes, list), 'invalid nodes input:str or list'
            output_nodes_list += output_nodes
        else:
            output_nodes_list += self.output

        if not isinstance(input_data, list):
            input_data = [input_data]
        if self.pre_processing_fn:
            input_data = [self.pre_processing_fn(item) for item in input_data]

        feed_dict = {key: value for key, value in zip(self.input, input_data)}
        result = self.sess.run(output_nodes_list, feed_dict=feed_dict)
        if self.post_processing_fn:
            result = [self.post_processing_fn(item) for item in result]
        if len(result) == 1:
            result = result[0]
        return result