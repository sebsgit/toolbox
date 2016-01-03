CC=gcc
CFLAGS+=-std=c99 -Wall -pedantic
LIBS+=
LDFLAGS+=
CORE_OBJ=
RM=rm -f
TARGET=a.out

all: main

main: $(CORE_OBJ) mempool.c test.c
	$(CC) $(CFLAGS) $(LIBS) $(LDFLAGS) $(CORE_OBJ) -pg mempool.c test.c -o $(TARGET)

clean:
	$(RM) $(CORE_OBJ) 2>/dev/null
distclean:
	$(RM) $(CORE_OBJ) 2>/dev/null
	$(RM) $(TARGET) 2>/dev/null
