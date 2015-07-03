#!/bin/bash
nvcc -ptx kernel.cu -o kernel.ptx
g++ -std=c++11 main.cpp -ldl -Wall -pthread
