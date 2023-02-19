cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)

project(
  paddlefx
  VERSION 0.1
  DESCRIPTION "paddlefx"
  LANGUAGES C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/paddlefx/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/paddlefx/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/paddlefx/)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

python3_add_library(_eval_frame MODULE WITH_SOABI src/paddlefx/_eval_frame.c)

install(TARGETS _eval_frame LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
