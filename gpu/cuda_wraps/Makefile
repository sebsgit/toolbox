CC=g++
CFLAGS+=-std=c++11 -Wall -Werror
LIBS+=
LDFLAGS+=-ldl -pthread
CORE_OBJ=
RM=rm -f
TARGET=a.out
CUDA_SRC=kernel.cu

all: main

main: $(CORE_OBJ) main.cpp $(CUDA_SRC) cuwrap.cpp
	$(CC) $(CFLAGS) $(CORE_OBJ) cuwrap.cpp main.cpp -o $(TARGET) $(LIBS) $(LDFLAGS)

kernel.ptx: kernel.cu
	nvcc -ptx kernel.cu -o kernel.ptx -arch=sm_20

clean:
	$(RM) $(CORE_OBJ) 2>/dev/null
distclean:
	$(RM) $(CORE_OBJ) 2>/dev/null
	$(RM) $(TARGET) 2>/dev/null
