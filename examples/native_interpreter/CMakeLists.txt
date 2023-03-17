cmake_minimum_required(VERSION 3.10)
project(myinterpreter CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v2.10.4
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(pybind11)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3_EXECUTABLE = ${Python3_EXECUTABLE}")

# find libpaddle
execute_process(
  COMMAND
    ${Python3_EXECUTABLE} -c
    "import paddle;import pathlib;print(pathlib.Path(paddle.__file__).parent, end='')"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE PADDLE_LIBRARY)
message(STATUS "PADDLE_LIBRARY = ${PADDLE_LIBRARY}")

include_directories(${PADDLE_LIBRARY}/include)
set(link_libraries ${PADDLE_LIBRARY}/fluid/libpaddle.so)

python3_add_library(myinterpreter MODULE WITH_SOABI interpreter.cc)
target_link_libraries(myinterpreter PRIVATE ${link_libraries} pybind11::embed)
