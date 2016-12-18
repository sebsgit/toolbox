cmake_minimum_required(VERSION 2.8)

project(huffmanCoding)
set(CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "-Wall -Werror -pedantic")
add_executable(${PROJECT_NAME} "main.cpp")
