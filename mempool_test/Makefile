CC=gcc
CFLAGS+=-std=c99 -Wall -pedantic
LIBS+=
LDFLAGS+= -Wl,--large-address-aware
CORE_OBJ=
RM=rm -f
TARGET=a.out

BTREE_DIR=../btree
INCLUDE_DIR+=$(BTREE_DIR)
SOURCES=mempool.c test.c $(BTREE_DIR)/btree.c

all: main

main: $(CORE_OBJ) $(SOURCES) $(BTREE_DIR)/btree.h
	$(CC) -I $(INCLUDE_DIR) $(CFLAGS) $(LIBS) $(LDFLAGS) $(CORE_OBJ) -pg $(SOURCES) -o $(TARGET)

clean:
	$(RM) $(CORE_OBJ) 2>/dev/null
distclean:
	$(RM) $(CORE_OBJ) 2>/dev/null
	$(RM) $(TARGET) 2>/dev/null
