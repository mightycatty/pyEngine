# @Time    : 2021/12/23 2:55 PM
# @Author  : Heshuai
# @Email   : heshuai.sec@gmail.com
import os

import onnx
from onnxsim import simplify


def onnx_graph_optimize(onnx_dir):
    try:
        model_proto = onnx.load(onnx_dir)
        model_proto, check = simplify(model_proto, skip_fuse_bn=False)
        assert check, "Simplified ONNX model could not be validated"
        output_name = onnx_dir.replace('.onnx', '_opt.onnx')
        onnx.save(model_proto, output_name)
    except Exception as e:
        print(e)
        return None
    return output_name


def convert_pb_onnx_to_mnn(frozen_pb_dir_or_onnx, fp16=True):
    """
     Usage:
      1. pip install -U MNN
      2. call this fn
    :param frozen_pb_dir_or_onnx:
    fp16: false if you want ot do quan later
    :return:
    """
    try:
        from MNN.tools.mnnconvert import Tools
    except ImportError:
        print('pip install -U MNN and try again')
    assert isinstance(frozen_pb_dir_or_onnx, str) and os.path.exists(frozen_pb_dir_or_onnx), \
        'invalid file or not exit/{}'.format(frozen_pb_dir_or_onnx)
    # framework index
    TF = 0
    ONNX = 2
    TFLITE = 4
    valid_format = {'PB': 0, 'ONNX': 2}  # mnn official framework index
    or_name = os.path.splitext(frozen_pb_dir_or_onnx)[0]
    file_format = os.path.splitext(frozen_pb_dir_or_onnx)[-1][1:]
    assert file_format.upper() in valid_format.keys(), 'invalid model format, pb/onnx are supported'
    framework_type = valid_format[file_format.upper()]
    output_name = or_name + '.mnn'
    # TODO: this might crush for unknown reason, use cmd instead, refer:https://www.yuque.com/mnn/cn/model_convert#ldDS5
    # Tools.mnnconvert(output_name, frozen_pb_dir_or_onnx, framework_type, fp16, "NA.mnn", 0, False, '', 'biz')
    cmd_str = r'MNNConvert -f ONNX --modelFile {} --MNNModel {} --bizCode biz'.format(frozen_pb_dir_or_onnx, output_name)
    if fp16:
        cmd_str += ' --fp16'
    os.system(cmd_str)
    return True


if __name__ == '__main__':
    onnx_dir = r'/Users/shuai.he/Projects/shopee-semantic-segmentation/semseg/models/ShuffleV2Seg_opt.onnx'
    convert_pb_onnx_to_mnn(onnx_dir)
