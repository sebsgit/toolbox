#!/bin/bash
set -e

OUT_FILE=sanitize.log

rm -f $OUT_FILE

clang++-9 -std=c++14 main.cpp -Wall -Werror -g -ggdb -pedantic -Og -fsanitize=address -o cmsg_asan
./cmsg_asan | tee -a $OUT_FILE
rm cmsg_asan

clang++-9 -std=c++14 main.cpp -Wall -Werror -g -ggdb -pedantic -Og -fsanitize=thread -o cmsg_tsan
./cmsg_tsan | tee -a $OUT_FILE
rm cmsg_tsan

clang++-9 -std=c++14 main.cpp -Wall -Werror -g -ggdb -pedantic -Og -fsanitize=undefined -o cmsg_ubsan
./cmsg_ubsan | tee -a $OUT_FILE
rm cmsg_ubsan
