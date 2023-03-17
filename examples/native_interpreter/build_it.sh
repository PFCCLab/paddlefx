cmake \
  -DPython3_EXECUTABLE=$(which python) \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
  -S. -Bbuild -G Ninja

cmake --build build --target all

# optional
# stubgen \
#   -m build.myinterpreter \
#   -o .
