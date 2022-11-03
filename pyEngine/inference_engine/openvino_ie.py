import logging
import os
import sys

from openvino.inference_engine import IENetwork, IECore

CPU_EXTENSION = r'C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\inference_engine\bin\intel64\Release' \
                r'\cpu_extension_avx2.dll'


class InferenceWithOpenvino:
    def __init__(self, model_file,
                 device='CPU',
                 pre_processing_fn=None,
                 post_processing_fn=None,
                 **kwargs):
        self.model_xml = model_file
        self.device = device
        logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
        self.pre_processing_fn = pre_processing_fn
        self.post_processing_fn = post_processing_fn
        self._model_init()

    def _model_init(self):
        model_bin = os.path.splitext(self.model_xml)[0] + ".bin"
        logging.info("Creating Inference Engine")
        ie = IECore()
        found_device = ie.available_devices
        logging.info("found devices:\n{}".format(found_device))
        ie.add_extension(CPU_EXTENSION, "CPU")
        # Read IR
        logging.info("Loading network files:\n\t{}\n\t{}".format(self.model_xml, model_bin))
        net = IENetwork(model=self.model_xml, weights=model_bin)
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))
        # resize network
        # net.reshape({self.input_blob: (1, 3, 256, 256)})
        if "CPU" in self.device:
            supported_layers = ie.query_network(net, "CPU")
            not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                logging.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                              format(self.device, ', '.join(not_supported_layers)))
                logging.error(
                    "Please try to specify cpu extensions library path in sample's command line parameters using -l "
                    "or --cpu_extension command line argument")
                sys.exit(1)
        assert len(net.inputs.keys()) == 1, "Sample supports only single input topologgingies"
        assert len(net.outputs) == 1, "Sample supports only single output topologgingies"
        logging.info("Preparing input blobs")
        net.batch_size = 1
        # Loading model to the plugin
        logging.info("Loading model to the plugin")
        config = {}
        # config['CPU_THREADS_NUM'] = '1'
        # config['CLDNN_PLUGIN_PRIORITY'] = '0'
        config = None
        self.exec_net = ie.load_network(network=net, device_name=self.device, config=config)

    def predict(self, input_data):
        if self.pre_processing_fn:
            input_data = self.pre_processing_fn(input_data)
        result = self.sess.run([self.output], feed_dict={self.input: input_data})[0]
        if self.post_processing_fn:
            result = self.post_processing_fn(result)
        return result
