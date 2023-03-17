  cmake -Bbuild -DPADDLE_LIBRARY=$(python -c "import paddle, os, inspect; print(os.path.dirname(inspect.getsourcefile(paddle)))") -DCMAKE_PREFIX_PATH=$(python -c "import pybind11, os, inspect; print(os.path.dirname(inspect.getsourcefile(pybind11)))")/..
  make -C build
