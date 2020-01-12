#ifndef _CLC_ENCRYPT_H__
#define _CLC_ENCRYPT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include "clc_aes.h"

typedef enum {
	CLC_AES_128,
	CLC_AES_192,
	CLC_AES_256
} clc_cipher_type;

extern void clc_aes_init_key(clc_aes_key* key, const unsigned char* data, const ssize_t data_len, const clc_cipher_type type);
extern void clc_aes_encrypt(unsigned char* output, const unsigned char* input, const ssize_t data_len, const clc_aes_key key, const clc_cipher_type type);
extern void clc_aes_decrypt(unsigned char* output, const unsigned char* input, const ssize_t data_len, const clc_aes_key key, const clc_cipher_type type);

#ifdef __cplusplus
}
#endif

#endif
