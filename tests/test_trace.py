from __future__ import annotations

import unittest

import paddle

import paddlefx


class TestFx(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.models_to_track = [
            (paddle.vision.models.resnet18(), paddle.randn([2, 3, 224, 224])),
            (paddle.vision.models.alexnet(), paddle.randn([2, 3, 224, 224])),
            (paddle.vision.models.densenet121(), paddle.randn([2, 3, 224, 224])),
            (paddle.vision.models.googlenet(), paddle.randn([2, 3, 224, 224])),
            (paddle.vision.models.inception_v3(), paddle.randn([2, 3, 299, 299])),
            (paddle.vision.models.mobilenet_v2(), paddle.randn([2, 3, 224, 224])),
            (
                paddle.vision.models.shufflenet_v2_swish(),
                paddle.randn([2, 3, 224, 224]),
            ),
            (paddle.vision.models.squeezenet1_0(), paddle.randn([2, 3, 224, 224])),
            (paddle.vision.models.vgg11(), paddle.randn([2, 3, 224, 224])),
            (paddle.vision.models.wide_resnet101_2(), paddle.randn([2, 3, 224, 224])),
        ]

    def tearDown(self):
        super().tearDown()

    def test_trace(self):
        for model, input_example in self.models_to_track:
            print(f"tracing model: {type(model).__name__}")
            traced_model = paddlefx.symbolic_trace(model)
            paddle.seed(1234)
            orig_output = model(input_example)
            paddle.seed(1234)
            traced_output = traced_model(input_example)

            # some nets, e.g.: googlenet, return a list of tensors
            orig_ret_list = (
                list(orig_output)
                if isinstance(orig_output, (list, tuple))
                else [orig_output]
            )
            traced_ret_list = (
                [*traced_output]
                if isinstance(traced_output, (list, tuple))
                else [traced_output]
            )

            self.assertEqual(
                len(orig_ret_list),
                len(traced_ret_list),
                f"model: {type(model).__name__} failed",
            )

            for i, o in enumerate(traced_ret_list):
                self.assertTrue(
                    paddle.allclose(orig_ret_list[i], traced_ret_list[i]),
                    f"model: {type(model).__name__} failed",
                )
