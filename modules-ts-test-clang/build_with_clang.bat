clang++ -fmodules-ts -std=c++2a --precompile module_interface.cppm -o test.pcm
clang++ -fmodules-ts -std=c++2a -c test.pcm -o module_interface.o
clang++ -fmodules-ts -std=c++2a -c module_implementation.cpp -o module_implementation.o -fmodule-file=test.pcm
clang++ -fmodules-ts -std=c++2a main.cpp module_interface.o module_implementation.o -fprebuilt-module-path=.