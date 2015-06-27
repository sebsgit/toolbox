#!/bin/sh

#
# print configuration related to libc and gcc tests
#

uname -risp
ldd --version | awk '{if(NR==1)print;}'
ld -v
make -v | awk '{if(NR==1) print ;}'
echo "DejaGnu:"
runtest -V
