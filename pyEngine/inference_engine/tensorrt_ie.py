import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)  # disable nasty future warning in tensorflow and numpy

import os
import tensorrt as trt
import logging
import uff
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import sys

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)  # global trt logger setting


class TensorrtBuilder:

    @staticmethod
    def _item_to_list(item):
        if not isinstance(item, list):
            if item:
                item = [item]
        return item

    @staticmethod
    def _GiB(val):
        return val * 1 << 30

    @staticmethod
    def _create_optimization_profile(builder, config, input_name, input_shape, batch_size=None):
        """
        required for mode with dynamic shape, call build_engine(network, config) instead of build_cuda_engine(network)
        :param builder:
        :param config:
        :param input_name: name of input nodes
        :param input_shape: ignore batch dim if batch_size is None
        :param batch_size: none for explict batch dim network
        :return: None, alteration is done to 'config' obj
        """
        profile = builder.create_optimization_profile()
        # Fixed explicit batch in input shape
        if batch_size is None:
            batch_size = input_shape[0]
            shape = input_shape[1:]
        # Dynamic explicit batch
        elif input_shape[0] == -1:
            shape = input_shape[1:]
        # Implicit Batch
        else:
            shape = input_shape

        min_batch = batch_size
        opt_batch = batch_size
        max_batch = batch_size
        profile.set_shape(input_name, min=(min_batch, *shape), opt=(opt_batch, *shape), max=(max_batch, *shape))
        config.add_optimization_profile(profile)

    @staticmethod
    def _save_engine(engine, dump_name) -> bool:
        dump_name = '{}.engine'.format(dump_name) if '.engine' not in dump_name else dump_name
        with open(dump_name, 'wb') as f:
            f.write(engine.serialize())
        return True

    @staticmethod
    def _load_engine(trt_runtime, engine_path):
        engine_path = '{}.engine'.format(engine_path) if '.engine' not in engine_path else engine_path
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    @staticmethod
    def _build_engine(network,
                      builder,
                      explicit_batch_dim=False,
                      max_batch_size=1,
                      max_workspace_size=1 << 30,
                      mix_precision='fp32',
                      calib=None):
        # dynamic shape building config with explict_batch_size = True
        if explicit_batch_dim:
            config = builder.create_builder_config()
            config.max_workspace_size = max_workspace_size
            if mix_precision == 'int8':
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = calib
            if mix_precision == 'fp16':
                config.set_flag(trt.BuilderFlag.FP16)
            input_shape = network.get_input(0).shape
            input_name = network.get_input(0).name
            TensorrtBuilder._create_optimization_profile(builder, config, input_name, input_shape, None)
            built_engine = builder.build_engine(network, config)
        else:
            builder.max_batch_size = max_batch_size
            builder.max_workspace_size = max_workspace_size
            if mix_precision == 'fp16':
                builder.fp16_mode = True
            if mix_precision == 'int8':
                builder.int8_mode = True
                builder.int8_calibrator = calib
            built_engine = builder.build_cuda_engine(network)
        return built_engine

    @staticmethod
    def _pb_uff_parser(pb_dir,
                       network,
                       input_node_names,
                       input_node_shapes,
                       output_node_names):
        parser = trt.UffParser()
        # parse network
        for input_node_name, input_node_shape in zip(input_node_names, input_node_shapes):
            parser.register_input(input_node_name, input_node_shape)
        for output_node_name in output_node_names:
            parser.register_output(output_node_name)
        uff_buffer = uff.from_tensorflow_frozen_model(frozen_file=pb_dir, output_nodes=output_node_names,
                                                      output_filename='buffer.uff', text=False,
                                                      debug_mode=True)
        parser.parse_buffer(uff_buffer, network)
        os.remove('buffer.uff')
        return network

    @staticmethod
    def build_engine_from_pb_or_onnx(model_file,
                                     input_node_names=None,
                                     input_node_shapes=None,
                                     output_node_names=None,
                                     explicit_batch_dim=False,
                                     max_batch_size=1,
                                     max_workspace_size=1 << 30,
                                     mix_precision='fp16',
                                     logger_level='verbose',
                                     calib=None):
        def _assertion():
            name, model_type = tuple(os.path.splitext(model_file))
            assert model_type in ['.pb', '.onnx'], 'invalid model format:{}/(pb-onnx)'.format(model_type)
            if model_type == '.pb':
                assert input_node_names and input_node_shapes and output_node_names, \
                    'input nodes names/shapes and output names are required for parsing .pb'
            assert mix_precision in ['fp16', 'fp32', 'int8'], 'invalid mix precision"{}/{}'. \
                format(mix_precision, ['fp16', 'fp32', 'int8'])
            if mix_precision == 'int8':
                assert calib is not None, 'calibrator is required for int8 mode'
            valid_logger_level = ['verbose', 'error']
            assert logger_level in valid_logger_level, 'valid log level:{}'.format(valid_logger_level)

        def _trt_logger():
            cmd_str = 'trt_logger = trt.Logger(trt.Logger.{})'.format(logger_level.upper())
            exec(cmd_str)
        _assertion()
        _trt_logger()

        logger.info('building engine from:{}'.format(model_file))
        logger.info('explict batch dim:{}'.format(explicit_batch_dim))
        name, model_type = tuple(os.path.splitext(model_file))

        # force explicit batch dim flag for onnx model, 7.0.0 only supports parsing onnx with explicit_batch flag
        if os.path.splitext(model_file)[-1] == '.onnx':
            explicit_batch_dim = 1
            logger.info('forcing explicit batch flag for onnx model, '
                        '7.0.0 only supports parsing onnx with explicit_batch flag')

        # initialize builder
        builder = trt.Builder(TRT_LOGGER)
        network_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) if explicit_batch_dim else 0
        network = builder.create_network(network_flag)
        # parse network
        if model_type == '.pb':
            input_node_names = TensorrtBuilder._item_to_list(input_node_names)
            output_node_names = TensorrtBuilder._item_to_list(output_node_names)
            network = TensorrtBuilder._pb_uff_parser(model_file, network, input_node_names, input_node_shapes,
                                                     output_node_names)
        else:
            parser = trt.OnnxParser(network, TRT_LOGGER)
            with open(model_file, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error('ERROR: Failed to parse the ONNX file: {}'.format(model_file))
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    sys.exit(1)
        # build engine
        built_engine = TensorrtBuilder._build_engine(network, builder, explicit_batch_dim, max_batch_size,
                                                     max_workspace_size,
                                                     mix_precision, calib)
        if built_engine:
            logger.info('engine built!')
            TensorrtBuilder._save_engine(built_engine, name)
            return built_engine
        else:
            logger.error('fail to build engine!')
            return False


class InferenceWithTensorRT:
    def __init__(self, model_file, pre_processing_fn=None, post_processing_fn=None, force_rebuild=False, **kwargs):
        self.model_dir = model_file
        self.pre_processing_fn = pre_processing_fn
        self.post_processing_fn = post_processing_fn
        self.kwargs = kwargs
        self.force_rebuild = force_rebuild
        self._engine_init()
        self._context_init()

    def _engine_init(self):
        """
        load a engine buffer or buid a new one
        :return: a trt engine obj
        """
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        self.trt_engine = None
        engine_file = os.path.splitext(self.model_dir)[0] + '.engine'
        if not os.path.exists(engine_file) or self.force_rebuild:
            print('no built engine found, building a new one...')
            model_type = os.path.splitext(self.model_dir)[-1]
            valid_model_format = ['.pb', '.onnx']
            assert model_type in valid_model_format, 'provided model is invalid:{}/{}'.format(model_type,
                                                                                              valid_model_format)
            self.trt_engine = TensorrtBuilder.build_engine_from_pb_or_onnx(self.model_dir, **self.kwargs)
        else:
            print('loading built engine:{}...'.format(engine_file))
            self.trt_engine = TensorrtBuilder._load_engine(self.trt_runtime, engine_file)

    def _context_init(self):
        volume = trt.volume(self.trt_engine.get_binding_shape(0)) * self.trt_engine.max_batch_size
        self.input_dtype = trt.nptype(self.trt_engine.get_binding_dtype(0))
        self.host_input = cuda.pagelocked_empty(volume, dtype=self.input_dtype)
        volume = trt.volume(self.trt_engine.get_binding_shape(1)) * self.trt_engine.max_batch_size
        dtype = trt.nptype(self.trt_engine.get_binding_dtype(1))
        self.host_output = cuda.pagelocked_empty(volume, dtype=dtype)
        # Allocate device memory for inputs and outputs.
        self.cuda_input = cuda.mem_alloc(self.host_input.nbytes)
        self.cuda_output = cuda.mem_alloc(self.host_output.nbytes)
        self.context = self.trt_engine.create_execution_context()
        self.context.active_optimization_profile = 0
        self.stream = cuda.Stream()

    def predict(self, input_data):
        """
        predict with async api
        data -> cpu -> GPU -> cpu
        :param input_data:
        :param kwargs:
        :return:
        """
        if self.pre_processing_fn is not None:
            input_data = self.pre_processing_fn(input_data)
        if str(input_data.dtype) != self.input_dtype.__name__:
            logging.warning('dtype of input data:{} is not compilable with engine input:{}, enforcing dtype convertion'
                            .format(str(input_data.dtype), self.input_dtype.__name__))
            input_data = self.input_dtype(input_data)
        # input data -> cpu
        np.copyto(self.host_input, input_data.ravel())
        # cpu -> gpu
        cuda.memcpy_htod_async(self.cuda_input, self.host_input, self.stream)
        # Run inference. difference execution api by the way the engine built(implicit/explicit batch size)
        if self.trt_engine.has_implicit_batch_dimension:
            self.context.execute_async(bindings=[int(self.cuda_input), int(self.cuda_output)],
                                       stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(bindings=[int(self.cuda_input), int(self.cuda_output)],
                                          stream_handle=self.stream.handle)
        # gpu -> cpu.
        cuda.memcpy_dtoh_async(self.host_output, self.cuda_output, self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        output = self.host_output
        if self.post_processing_fn is not None:
            output = self.post_processing_fn(output)
        # Return the host output.
        return output


class CustomEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """
    simple calibrator passed to builder for building a int8 engine.
    """

    def __init__(self, data_gen,  # a python generator, each yield return a batch of x(N, C)/ (y is not required)
                 cache_file,  # calibrator cache file name, str
                 batch_size=8,
                 input_shape_wo_batch_dim=(4, 1024, 1024)):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file
        self.batch_size = batch_size
        self.data_gen = data_gen
        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(np.ones((batch_size, *input_shape_wo_batch_dim), dtype=np.float32).nbytes)
        self.batch_count = 0
        self.input_shape = input_shape_wo_batch_dim

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            batch_data = next(self.data_gen)
            assert batch_data.shape == (self.batch_size, *self.input_shape), 'date batch size is invalid'
            cuda.memcpy_htod(self.device_input, batch_data.ravel())
            # if self.batch_count % 1 == 0:
            logger.info("Calibrating batch {:}, containing {:} images".format(self.batch_count, self.batch_size))
            self.batch_count += 1
            return [int(self.device_input)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            logger.info('calibrate cache found')
            with open(self.cache_file, "rb") as f:
                return f.read()
        else:
            logger.info('no calibrate cache found')
            return

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


