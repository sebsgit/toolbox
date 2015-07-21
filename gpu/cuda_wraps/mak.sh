#!/bin/bash
nvcc -ptx kernel.cu -o kernel.ptx -arch=sm_20
g++ -std=c++11 cuwrap.cpp main.cpp -ldl -Wall -pthread
