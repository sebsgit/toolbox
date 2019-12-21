#!/bin/bash
clang++-9 -std=c++14 main.cpp -Wall -Werror -g -ggdb -pedantic -Og -o cmsg_test && ./cmsg_test
