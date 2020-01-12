#ifndef _CLC_ENCRYPT_AES_H_
#define _CLC_ENCRYPT_AES_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "clc_bytes.h"

typedef struct {
	unsigned char b[176];
} clc_aes_key_128;

extern void clc_init_key_16( clc_aes_key_128 * key, const unsigned char * data, int data_len );
extern void clc_expand_key_16( clc_aes_key_128 * key );
extern void clc_print_key_16( const clc_aes_key_128 * key );

extern void clc_encrypt_16( clc_bytes_16 * x, const clc_aes_key_128 * key );
extern void clc_decrypt_16( clc_bytes_16 * x, const clc_aes_key_128 * key );

typedef struct {
	unsigned char b[208];
} clc_aes_key_192;

extern void clc_init_key_24( clc_aes_key_192 * key, const unsigned char * data, int data_len );
extern void clc_expand_key_24( clc_aes_key_192 * key );
extern void clc_print_key_24( const clc_aes_key_192 * key );

extern void clc_encrypt_24( clc_bytes_16 * x, const clc_aes_key_192 * key );
extern void clc_decrypt_24( clc_bytes_16 * x, const clc_aes_key_192 * key );

typedef struct {
	unsigned char b[240];
} clc_aes_key_256;

extern void clc_init_key_32( clc_aes_key_256 * key, const unsigned char * data, int data_len );
extern void clc_expand_key_32( clc_aes_key_256 * key );
extern void clc_print_key_32( const clc_aes_key_256 * key );

extern void clc_encrypt_32( clc_bytes_16 * x, const clc_aes_key_256 * key );
extern void clc_decrypt_32( clc_bytes_16 * x, const clc_aes_key_256 * key );

typedef union {
	clc_aes_key_256 key256;
	clc_aes_key_192 key192;
	clc_aes_key_128 key128;
} clc_aes_key;

#ifdef __cplusplus
}
#endif

#endif