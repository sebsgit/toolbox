#!/bin/bash
g++-6 -Os -std=c++14 -I. main.cpp -Wall -o test
if [[ $? -ne 0 ]]; then
	echo "g++ build failed!"
	exit 1
fi
./test
echo "---"
clang++-3.8 -std=c++1z -I. main.cpp -ftemplate-depth=900 -Wall -Os
if [[ $? -ne 0 ]]; then
	echo "clang++ build failed!"
	exit 2
fi
./test
