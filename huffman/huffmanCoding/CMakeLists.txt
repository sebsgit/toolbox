cmake_minimum_required(VERSION 2.8)

project(huffmanCoding)
set(CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "-Wall -Werror -pedantic")
include_directories(${CMAKE_SOURCE_DIR})
add_executable(${PROJECT_NAME} "main.cpp")
