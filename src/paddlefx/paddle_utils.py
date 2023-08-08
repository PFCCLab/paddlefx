from __future__ import annotations

import types

import paddle

Tensor = paddle.Tensor
TensorType = type(paddle.Tensor())


def skip_paddle_frame(frame: types.FrameType) -> bool:
    # NOTE: skip paddle internal code
    if frame.f_code.co_filename.endswith('paddle/fluid/dygraph/math_op_patch.py'):
        return True
    elif frame.f_code.co_filename.endswith('paddle/fluid/framework.py'):
        return True
    elif frame.f_code.co_filename.endswith('paddle/tensor/to_string.py'):
        return True
    elif frame.f_code.co_filename.endswith('fluid/dygraph/varbase_patch_methods.py'):
        return True
    elif frame.f_code.co_name == 'in_dygraph_mode':
        return True
    else:
        return False


def skip_paddle_filename(filename: str) -> bool:
    # TODO: use regex
    if '/paddle/fluid/' in filename:
        return True
    elif '/paddle/nn/' in filename:
        return True
    elif '/paddle/tensor/' in filename:
        return True
    else:
        return False
