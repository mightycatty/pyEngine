"""tensorflow graph toolkit
"""
import logging
import os
from collections import Counter
from functools import wraps

import tensorflow as tf
from tensorflow.python.keras import backend as K

from onnx_graph_toolkit import convert_pb_onnx_to_mnn

logger = logging.getLogger('tf_graph_toolkit')
logger.setLevel(logging.INFO)


# ================================= commonuse utility ======================================
def _graph_pb_graphdef(input_obj):
    """get type from an input-obj, pb or graph or graphdef, otherwise none"""
    valid_format = ['pb', 'graph', 'graphdef']
    if isinstance(input_obj, str):
        return 'pb'
    elif isinstance(input_obj, object):
        type = input_obj.__class__.__name__
        if type.lower() in valid_format:
            return type
    return None


def read_pb(graph_filepath):
    """read a .pb and return a graph_def obj"""
    try:
        with tf.gfile.GFile(graph_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def
    except Exception as e:
        logger.error('Pb reading error:{}').format(e)
        exit(0)


def _get_graph_def(pb_or_graph):
    """return a graphdef obj with a .pb file or graph obj input"""
    type = _graph_pb_graphdef(pb_or_graph)
    assert type, 'invalid input for getting a graph def'
    if type == 'pb':
        assert os.path.exists(pb_or_graph), 'pb file not exist:{}'.format(pb_or_graph)
        graph_def = read_pb(pb_or_graph)
    elif type == 'Graph':
        graph_def = pb_or_graph.as_graph_def()
    elif type == 'GraphDef':
        graph_def = pb_or_graph
    else:
        logger.error('unknown error when getting graph_def')
        exit(0)
    return graph_def


def get_graphdef_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        graph_input = args[0]  # assume the first input of func is pb_or_graph_def
        graph_def = _get_graph_def(graph_input)
        args = tuple([graph_def] + list(args[1:]))
        result = func(*args, **kwargs)
        return result

    return wrapper


def get_node_by_name_in_graphdef(graph_def, node_name):
    """get node by name in graphdef, return a nodedef if exist otherwise None"""
    # inputs detection
    for node in graph_def.node:
        if node.name == node_name:
            return node
    return None


def get_node_by_optype_in_graphdef(graph_def, optype_list):
    """get nodes by op type, return a list of nodedef if detected"""
    node_list = []
    optype_list = [optype_list] if isinstance(optype_list, str) else optype_list
    # inputs detection
    for node in graph_def.node:
        if node.op in optype_list:
            node_list.append(node_list)
    return node_list


# TODO
def get_node_shape(graph_def, op_name):
    node = get_node_by_name_in_graphdef(graph_def, op_name)
    assert node, 'node with name:{} not found'.format(op_name)


# ================================= graph analysis ======================================
# TODO: rewrite input_shape to fix shape internally
def calculate_flogs(graph_or_pb, input_tensor_name=None, input_shape=None, *args, **kwargs):
    """
    cal flops of a tensorflow graph or a frozen.pb
    usage sample:
        # 0. from graph obj
            # explicit batch size is require for a meaningful calculation under this circumstance
            # input_tensor = tklib.placeholder(shape=(1, h, w, c)) explicit batch size of 1
            calculate_flops(graph)
        # 1. from pb file
            calculate_flops(pb, input_tensor_name='old_input_tensor_name:0', (1, 512, 512, 3))
    :param graph_or_pb: tensorflow graph or a frozen pb
    :param input_tensor_name: your original input tensor name
    :param input_shape: input shape with explicit batchsize of 1: (1, h, w, c)
    :return:
    """
    if isinstance(graph_or_pb, str):
        assert input_tensor_name and input_shape, 'input_tensor_name and input_shape is required with a .pb input'
        graph_def = read_pb(graph_or_pb)
        new_input_tensor = tf.placeholder(dtype=tf.float32, shape=input_shape)
        tf.import_graph_def(graph_def, input_map={input_tensor_name: new_input_tensor})
        graph_or_pb = tf.get_default_graph()
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(graph=graph_or_pb,
                                run_meta=run_meta, cmd='scope', options=opts)

    return flops.total_float_ops


# TODO: TO TESTc
@get_graphdef_wrapper
def convert_pb_to_summary(input_path, output_dir=None, start_tensorboard=False, port=8000):
    if not output_dir:
        output_dir = input_path + ".summary"

    logging.info("load from %s", input_path)
    graph_def = read_pb(input_path)

    logging.info("save to %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with tf.Session() as sess:
        tf.import_graph_def(graph_def, name=os.path.basename(input_path).split('.')[0])
        train_writer = tf.summary.FileWriter(output_dir)
        train_writer.add_graph(sess.graph)
        train_writer.close()

    if start_tensorboard:
        logging.info("launch tensorboard")
        os.system("start tensorboard --logdir {} --port {}".format(output_dir, port))
        os.system("start http://localhost:{}".format(port))


# TODO: TO TEST
@get_graphdef_wrapper
def print_graph_stat(graph_def):
    op_stat = Counter()
    for node in graph_def.node:
        op_stat[node.op] += 1

    print("graph stat:")
    for op, count in sorted(op_stat.items(), key=lambda x: x[0]):
        print("\t%s = %s", op, count)


def _auto_inputs_outputs_detect(graph_def):
    """
    automatically detect inputs(nodes with op='Placeholder') and outputs(nodes without output edges) given a graph_def.
    Place note that this is not 100% safe, might yield wrong result, double check before carrying on
    :param graph_def:
    :return: inputs(list), outputs(list), eg. ['input_0:0', 'input_1:0], ['output:0']
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


# TODO: use model analyse from tensorflow official instead
def count_weights():
    import tensorflow as tf
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    # print(total_parameters)
    return total_parameters


# ================================= graph post-processing and optimization ======================================
def _freeze_session(session, output_node_names, keep_var_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_node_names, freeze_var_names)
        return frozen_graph


def freeze_sess_to_constant_pb(sess, export_name=None, output_node_names=None, as_text=False,
                               dump_result=False, *args, **kwargs):
    """
     output a constant graph for inference and test from a active tklib session
    keep in mind that usually a session in tensorflow if full of duplicate and useless stuff, clean it up before export
    known bug:
        sometime frozen pb only consists of constant node without edges.
    :param sess: activate tklib session with graph and initialized variables
    :param export_name: export name of the .pb file
    :param output_node_names: name of output nodes in graph, auto detect if none given(not 100% safe): node names, not tensor with ':0'
    :param as_text:
    :param dump_result:
    :param args:
    :param kwargs:
    :return: frozen graph_def or False
    """

    def _freeze_session(session, keep_var_names=None, clear_devices=True):
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                session, input_graph_def, output_node_names, freeze_var_names)
            return frozen_graph

    try:
        if output_node_names is None:
            _, output_node_names = _auto_inputs_outputs_detect(sess.graph.as_graph_def())
        output_node_names = [output_node_names] if not isinstance(output_node_names, list) else output_node_names
        output_node_names = [item.strip(':0') for item in output_node_names]
        frozen_graph = _freeze_session(sess)
        if dump_result:
            tf.io.write_graph(frozen_graph, '.', export_name + '.pb', as_text=as_text)
        return frozen_graph
    except Exception as e:
        print(e)
        return False


def clean_graph_for_inference(graph_or_graph_def, input_node_names, output_node_names):
    """
    trim useless and training-relative nodes for inference.
    do mind that it's merely about graph cleanness, not graph level optimization
    :param graph_or_graph_def:
    :param input_node_names: name of input nodes, str or list
    :param output_node_names:
    :return: graph_def
    """
    from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
    # ================================ graph optimization ==================================
    input_node_names = [input_node_names] if type(input_node_names) is str else input_node_names
    output_node_names = [output_node_names] if type(output_node_names) is str else output_node_names
    input_node_names = [item.strip(':0') for item in input_node_names]
    output_node_names = [item.strip(':0') for item in output_node_names]
    placeholder_type_enum = tf.float32.as_datatype_enum
    if 'GraphDef' not in str(type(graph_or_graph_def)):
        graph_or_graph_def = graph_or_graph_def.as_graph_def()
    graph_def = optimize_for_inference(graph_or_graph_def, input_node_names, output_node_names, placeholder_type_enum)
    return graph_def


@get_graphdef_wrapper
def graph_optimization(frozen_pb_or_graph_def, input_names=None, output_names=None, transforms=None):
    """
    optimize graph for inference
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md
    do mind:
        1. output pb is not best for visualization
        2. constants folding is limit in tensorflow graph transforms, with explicit batch size 1 enables more constants folding,
            however still constants not folded.
        3. fix shape for input_tensor is recommanded
    :param frozen_pb_or_graph_def:
    :param input_names: str or list
    :param output_names: str or list
    :param transforms:
    :return: optimize graph def
    """
    try:
        from tensorflow.tools.graph_transforms import TransformGraph
        if transforms is None:
            transforms = [
                'remove_nodes(op=Identity)',
                # 'merge_duplicate_nodes', # not good for visualization
                'strip_unused_nodes',
                # 'remove_attribute(attribute_name=_class)',
                'fold_constants(ignore_errors=true)',
                'fold_batch_norms',
                # 'sort_by_execution_order',
                # 'fuse_convolutions',
                'remove_device',
                # 'quantize_nodes',
                # 'quantize_weights',
            ]
        if isinstance(input_names, str):
            input_names = [input_names]
        if isinstance(output_names, str):
            output_names = [output_names]
        if isinstance(frozen_pb_or_graph_def, str):
            graph_def = read_pb(frozen_pb_or_graph_def)
        else:
            graph_def = frozen_pb_or_graph_def
        if (input_names is None) and (output_names is None):
            input_names, output_names = _auto_inputs_outputs_detect(graph_def)
        optimized_graph_def = TransformGraph(graph_def,
                                             input_names,
                                             output_names,
                                             transforms)
        return optimized_graph_def
    except Exception as e:
        print('graph optimization error:{}'.format(e))
        print('maybe fix the shape of your input_tensor and try again')
        return False


# TODO: BUG, AVOID TO USE, use freeze_kears_model_to_pb_from_model_fn() instead
def freeze_keras_model_to_pb(tk_model, export_dir='.', export_name=None, optimize_graph=True, verbose=True):
    """
    not safe
    Known issue:
         run into 'SystemError: unknown opcode' if your keras model contains lambda layers
    :param tk_model:
    :param export_dir:
    :param export_name:
    :param optimize_graph:
    :param verbose:
    :return:
    """
    sess_or = K.get_session()
    weights = tk_model.get_weights()
    try:
        with tf.Graph().as_default() as graph, tf.Session(graph=graph).as_default() as sess:
            K.set_session(sess)
            K.set_learning_phase(0)
            new_model = tf.keras.models.clone_model(tk_model)
            new_model.trainable = False
            sess.run(tf.global_variables_initializer())
            new_model.set_weights(weights)
            input_names = [item.name for item in new_model.inputs]
            output_names = [item.name for item in new_model.outputs]
            graph = freeze_sess_to_constant_pb(sess, output_node_names=output_names)
            if export_name:
                export_name = export_name.strip('.pb')
            else:
                export_name = tk_model.name
            tf.io.write_graph(graph, export_dir, export_name + '.pb', as_text=False)
            if optimize_graph:
                graph = graph_optimization(graph, input_names=input_names, output_names=output_names)
                tf.io.write_graph(graph, export_dir, export_name + '.opt.pb')
            if verbose:
                print('frozen pb saved to:{}'.format(os.path.join(export_dir, export_name)))
    except Exception as e:
        print(e)
    finally:
        K.set_session(sess_or)  # rollback keras session
    return


# TODO: to test
def convert_frozen_pb_to_onnx(frozen_pb_or_graph_def, opset=10, tf_graph_optimization=True, input_shape=None, name=None):
    try:
        from tf2onnx.tfonnx import process_tf_graph, tf_optimize
        from tf2onnx import constants, logging, utils, optimizer
    except Exception as e:
        logger.error(e)
        logger.error('import tf2onnx error, "pip install tf2onnx"')
        exit(0)

    if isinstance(frozen_pb_or_graph_def, str):
        model_path = frozen_pb_or_graph_def
        output_dir = model_path.replace('.pb', '.onnx')
        graph_def = read_pb(frozen_pb_or_graph_def)
    else:
        model_path = 'graphdef_buffer'
        assert name, 'name should be give to export an .onnx when converting from a graphdef buffer'
        output_dir = '{}.onnx'.format(name)
        graph_def = frozen_pb_or_graph_def
    inputs, outputs = _auto_inputs_outputs_detect(graph_def)
    shape_override = {}
    if input_shape:
        assert isinstance(input_shape, list), 'input_shape item need to be list, each for dims of a input tensor'
        for idx, item in enumerate(input_shape):
            shape_override[inputs[idx]] = item
            # graph optimizatin with tf_graph_transform
    if tf_graph_optimization:
        graph_def = graph_optimization(graph_def)
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name='')
    with tf.Session(graph=tf_graph):
        onnx_graph = process_tf_graph(tf_graph,
                             continue_on_error=False,
                             target='',
                             opset=opset,
                             custom_op_handlers={},
                             extra_opset=[],
                             shape_override=shape_override,
                             input_names=inputs,
                             output_names=outputs,

                             inputs_as_nchw=None)
    # graph optimization with onnx optimizer
    onnx_graph = optimizer.optimize_graph(onnx_graph)
    model_proto = onnx_graph.make_model("converted from {}".format(model_path))
    # optimize with onnx-simplifier (pip install onnx-simplifier)
    from onnxsim import simplify
    model_proto, check = simplify(model_proto, skip_fuse_bn=False)

    assert check, "Simplified ONNX model could not be validated"

    # write onnx graph
    logger.info("")
    logger.info("Successfully converted TensorFlow model %s to ONNX", model_path)
    utils.save_protobuf(output_dir, model_proto)
    logger.info("ONNX model is saved at %s", output_dir)


# TODO: bugs, no .mnn saved after this scrip


# TODO: code stylish
def export_keras_model(model_fn,
                       weight_path,
                       input_shape,
                       export_path,
                       export_name,
                       graph_optimize=True,
                       export_tf_lite=False,
                       export_onnx=False,
                       onnx_opset=9,
                       export_mnn=False,
                       mnn_fp16=True,
                       batch_size=1,
                       debug=False):
    """
    由于tf keras的底层机制问题，最好不要用参数传递带参数的keras model对象，带参数的模型对象会有一个session，容易导致转pb错误
    注意：该方法导出的pb会多一个import前缀，如keras model下op名字为input, 则pb中为import/input:0
    :param model_with_weights:
    :param export_path:
    :param export_name:
    :return:
    """

    def _model_wrapper(model_fn, input_shape, model_name):
        input_tensor = tf.keras.layers.Input(input_shape, name='input', batch_size=batch_size)
        segmentation_output = model_fn(input_tensor)
        model = tf.keras.models.Model(input_tensor, segmentation_output, name=model_name)
        return model

    def _freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                session, input_graph_def, output_names, freeze_var_names)
            return frozen_graph

    if (not debug) and (weight_path is not None):
        assert os.path.exists(weight_path), 'weights not found'
    K.clear_session()
    K.set_learning_phase(0)  # all new operations will be in test mode from now on,
    # which is crucial for converting to tflite and a frozen pb
    model = _model_wrapper(model_fn, input_shape, export_name)
    if (not debug) and (weight_path is not None):
        model.load_weights(weight_path, by_name=True)
    with K.get_session() as sess:
        frozen_graph = _freeze_session(sess, output_names=[out.op.name for out in model.outputs])
        tf.io.write_graph(frozen_graph, export_path, export_name + '.pb', as_text=False)
        if graph_optimize:
            frozen_graph = graph_optimization(frozen_graph)
            tf.io.write_graph(frozen_graph, export_path, export_name + '_opt.pb', as_text=False)
        # convert to tflite
        if export_tf_lite:
            converter = tf.lite.TFLiteConverter.from_session(tf.keras.backend.get_session(), model.inputs,
                                                             model.outputs)
            tflite_model = converter.convert()
            save_dir = os.path.join(export_path, export_name + '.tflite')
            with open(save_dir, 'wb') as f:
                f.write(tflite_model)
    if export_onnx:
        convert_frozen_pb_to_onnx(frozen_graph, opset=onnx_opset, name=export_name, tf_graph_optimization=True)
    if export_mnn:
        convert_pb_onnx_to_mnn(export_name + '_opt.pb', fp16=mnn_fp16)
    return frozen_graph


if __name__ == '__main__':
    model_dir = r'/Users/shuai.he/Projects/shopee-semantic-segmentation/semseg/models/ShuffleV2_Human_Parsing.onnx'
    model_dir = convert_pb_onnx_to_mnn(model_dir, True)
    convert_pb_onnx_to_mnn(model_dir)
