#include "clc_encrypt.h"
#include <string.h>

void clc_aes_init_key(clc_aes_key* key, const unsigned char* data, const ssize_t data_len, const clc_cipher_type type) {
	switch (type) {
		case CLC_AES_128:
			clc_init_key_16(&key->key128, data, data_len);
			clc_expand_key_16(&key->key128);
		break;
		case CLC_AES_192:
			clc_init_key_24(&key->key192, data, data_len);
			clc_expand_key_24(&key->key192);
		break;
		case CLC_AES_256:
			clc_init_key_32(&key->key256, data, data_len);
			clc_expand_key_32(&key->key256);
		break;
		default:
			break;
	}
}

void clc_aes_encrypt(unsigned char* output, const unsigned char* input, const ssize_t data_len, const clc_aes_key key, const clc_cipher_type type) {
	ssize_t left = data_len;
	clc_bytes_16* input_ptr = (clc_bytes_16*)input;
	clc_bytes_16* output_ptr = (clc_bytes_16*)output;
	clc_bytes_16 buff;
	while (left >= 16) {
		left -= 16;
		memcpy(&buff, input_ptr, sizeof(buff));
		switch(type) {
			case CLC_AES_128: clc_encrypt_16(&buff, &key.key128); break;
			case CLC_AES_192: clc_encrypt_24(&buff, &key.key192); break;
			case CLC_AES_256: clc_encrypt_32(&buff, &key.key256); break;
			default: break;
		}
		memcpy(output_ptr, &buff, sizeof(buff));
		++output_ptr;
		++input_ptr;
	}
}

void clc_aes_decrypt(unsigned char* output, const unsigned char* input, const ssize_t data_len, const clc_aes_key key, const clc_cipher_type type) {
	ssize_t left = data_len;
	clc_bytes_16* input_ptr = (clc_bytes_16*)input;
	clc_bytes_16* output_ptr = (clc_bytes_16*)output;
	clc_bytes_16 buff;
	while (left >= 16) {
		left -= 16;
		memcpy(&buff, input_ptr, sizeof(buff));
		switch(type) {
			case CLC_AES_128: clc_decrypt_16(&buff, &key.key128); break;
			case CLC_AES_192: clc_decrypt_24(&buff, &key.key192); break;
			case CLC_AES_256: clc_decrypt_32(&buff, &key.key256); break;
			default: break;
		}
		memcpy(output_ptr, &buff, sizeof(buff));
		++output_ptr;
		++input_ptr;
	}
}
