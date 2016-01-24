CC=g++
CFLAGS+=-std=c++11 -Wall -Werror -pedantic
LIBS+=
LDFLAGS+=
CORE_OBJ=
RM=rm -f
TARGET=a.out

all: main

main: $(CORE_OBJ) main.cpp
	$(CC) $(CFLAGS) $(LIBS) $(LDFLAGS) $(CORE_OBJ) main.cpp -o $(TARGET)

clean:
	$(RM) $(CORE_OBJ) 2>/dev/null
distclean:
	$(RM) $(CORE_OBJ) 2>/dev/null
	$(RM) $(TARGET) 2>/dev/null
