CC=gcc
CFLAGS+=-Wall -Werror -pedantic
LIBS+=
LDFLAGS+=
DPSTR_PATH=../dpstring
INCLUDE+=-I../dpstring
CORE_OBJ=dpstring.o dpstringlist.o dpfileio.o inips.o
RM=rm -f
TARGET=a.out

all: main

main: $(CORE_OBJ) main.c
	$(CC) $(CFLAGS) $(LIBS) $(LDFLAGS) $(INCLUDE) $(CORE_OBJ) main.c -o $(TARGET)

dpstring.o: $(DPSTR_PATH)/dpstring.c
	$(CC) $(CFLAGS) $(LIBS) $(LDFLAGS) $(INCLUDE) $(DPSTR_PATH)/dpstring.c -c -o dpstring.o

dpstringlist.o: $(DPSTR_PATH)/dpstringlist.c
	$(CC) $(CFLAGS) $(LIBS) $(LDFLAGS) $(INCLUDE) $(DPSTR_PATH)/dpstringlist.c -c -o dpstringlist.o

dpfileio.o: $(DPSTR_PATH)/dpfileio.c
	$(CC) $(CFLAGS) $(LIBS) $(LDFLAGS) $(INCLUDE) $(DPSTR_PATH)/dpfileio.c -c -o dpfileio.o

inips.o: inips.c
	$(CC) $(CFLAGS) $(LIBS) $(LDFLAGS) $(INCLUDE) inips.c -c -o inips.o

clean:
	$(RM) $(CORE_OBJ) 2>/dev/null
distclean:
	$(RM) $(CORE_OBJ) 2>/dev/null
	$(RM) $(TARGET) 2>/dev/null
