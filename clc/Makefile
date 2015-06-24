DEL = rm
COPY = cp
CC = gcc
CFLAGS += -Wall -Werror -pedantic -msse -fPIC
LFLAGS +=
TARGET = test.exe
LIB_NAME = clc
LIB = lib/clc.dll

OBJECTS = build/clc_bytes.o \
		  build/clc_encrypt_16.o \
		  build/clc_encrypt_24.o \
		  build/clc_encrypt_32.o \
		  build/clc_encrypt_base.o	\
		  build/clc_md5.o	\
		  build/clc_sha1.o	\
		  build/clc_error.o	\
		  build/clc_file.o

all: $(TARGET) $(LIB)
	$(CC) main.c $(CFLAGS) -o $(TARGET) -Llib -l$(LIB_NAME)
	
$(TARGET): main.c $(LIB)

$(LIB): $(OBJECTS)
	$(CC) $(OBJECTS) -shared $(CFLAGS) -o $(LIB)
	$(COPY) $(LIB) .

build/clc_bytes.o : clc_bytes.c clc_bytes.h
	$(CC) clc_bytes.c -c $(CFLAGS) -o build/clc_bytes.o

build/clc_encrypt_16.o: clc_encrypt_16.h clc_encrypt_16.c
	$(CC) clc_encrypt_16.c -c $(CFLAGS) -o build/clc_encrypt_16.o

build/clc_encrypt_24.o: clc_encrypt_24.h clc_encrypt_24.c
	$(CC) clc_encrypt_24.c -c $(CFLAGS) -o build/clc_encrypt_24.o

build/clc_encrypt_32.o: clc_encrypt_32.h clc_encrypt_32.c
	$(CC) clc_encrypt_32.c -c $(CFLAGS) -o build/clc_encrypt_32.o

build/clc_encrypt_base.o: clc_encrypt_base.h clc_encrypt_base.c
	$(CC) clc_encrypt_base.c -c $(CFLAGS) -o build/clc_encrypt_base.o

build/clc_md5.o: clc_md5.h clc_md5.c
	$(CC) clc_md5.c -c $(CFLAGS) -o build/clc_md5.o

build/clc_sha1.o: clc_sha1.h clc_sha1.c
	$(CC) clc_sha1.c -c $(CFLAGS) -o build/clc_sha1.o

build/clc_error.o: clc_error.h clc_error.c
	$(CC) clc_error.c -c $(CFLAGS) -o build/clc_error.o

build/clc_file.o: clc_file.c clc_file.h
	$(CC) clc_file.c -c $(CFLAGS) -o build/clc_file.o

clean:
	$(DEL) $(OBJECTS)
	
distclean:
	$(DEL) $(OBJECTS)
	$(DEL) $(TARGET)
	$(DEL) $(LIB)
	$(DEL) $(LIB_NAME).dll
	
