CC=gcc
CFLAGS+=-std=c99 -Wall -Werror -pedantic
LIBS+=
LDFLAGS+=
CORE_OBJ=
RM=rm -f
TARGET=a.out

all: main

main: $(CORE_OBJ) main.c btree.c
	$(CC) $(CFLAGS) $(LIBS) $(LDFLAGS) $(CORE_OBJ) -O2 main.c btree.c -o $(TARGET)

clean:
	$(RM) $(CORE_OBJ) 2>/dev/null
distclean:
	$(RM) $(CORE_OBJ) 2>/dev/null
	$(RM) $(TARGET) 2>/dev/null
