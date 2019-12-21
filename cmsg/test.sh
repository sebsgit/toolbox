#!/bin/bash
set -e

clang++-9 -std=c++14 main.cpp -Wall -Werror -g -ggdb -pedantic -Og -o cmsg_test 2>&1 | tee compilation.log
./cmsg_test
