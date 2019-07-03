cmake_minimum_required(VERSION 3.1)

include(modules_experimental.cmake)

set(PROJECT_NAME "modules_test")
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)

add_executable(${PROJECT_NAME} "")

target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Werror -fmodules-ts -std=c++2a)

target_sources(${PROJECT_NAME} PRIVATE main.cpp)
cxx_module_generate(${PROJECT_NAME} test module_interface.cppm)
cxx_module_add_source(${PROJECT_NAME} test module_implementation.cpp)