trying to implement : https://github.com/pytorch/examples/tree/main/fx/native_interpreter

Steps to run:

- ./build_it.sh
- python use_interpreter.py

Some challenges remain:

- alignment of paddle::Tensor in C++ with paddle.Tensor in python, and seamlessly pass in/out between C++ and python
- infrastructure/utilities for such kind of application
- net parameters handling for real application.
- backward graph capture if target training.
