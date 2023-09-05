# paddlefx

paddlefx is an experimental project for building [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) Python IR.

## Community activities

- Tracking issue for PaddleSOT project: https://github.com/PaddlePaddle/PaddleSOT/issues/133
- Online tech discussion meeting: https://github.com/PaddlePaddle/PaddleSOT/discussions/149
- China open source software innovation challenge: https://github.com/PaddlePaddle/Paddle/issues/53172#paddlepaddle01

If you are interested in these activities, please send a email to ext_paddle_oss@baidu.com, we'll invite you into wechat group.

## Quick Start

```bash
pip install -r requirements_dev.txt

pip install -e .

pytest -n3 tests
```

## Explore its Capabilities

Here are some examples of what paddlefx can do:

- Capture and compile python bytecodes into a fx graph. See [this example](https://github.com/PFCCLab/paddlefx/blob/main/examples/targets/target_0_add.py).
- Trace ResNet and 10 other vision models. See [this example](https://github.com/PFCCLab/paddlefx/blob/main/examples/resnet_trace.py) and [this test](https://github.com/PFCCLab/paddlefx/blob/main/tests/test_trace.py).
- Edit fx graphs. See [this example](https://github.com/PFCCLab/paddlefx/blob/main/examples/graph_editing.py).
- Profile ResNet. See [this example](https://github.com/PFCCLab/paddlefx/blob/main/examples/fx_profiling.py).
- Demonstrate how to lower the fx graph IR to a native interpreter. See [this example](https://github.com/PFCCLab/paddlefx/tree/main/examples/native_interpreter).

## Contribution

This is a community driven project, maintenance of this project is on a "best effort" basis. The ideas and even lots of codes are borrowed from [pytorch 2.0](https://pytorch.org/get-started/pytorch-2.0/).

If you'd like to contribute, please feel free to raise a pull request, discuss in [issues](https://github.com/PFCCLab/paddlefx/issues) or [discussions](https://github.com/PFCCLab/paddlefx/discussions).


## test ci
