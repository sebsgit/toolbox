#ifndef _CLC_ENCRYPT_16_H_
#define _CLC_ENCRYPT_16_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "clc_encrypt_base.h"

extern void clc_init_key_16( clc_key_exp_16 * key, const unsigned char * data, int data_len );

extern void clc_expand_key_16( clc_key_exp_16 * key );

extern void clc_encrypt_16( clc_bytes_16 * x, clc_key_exp_16 * key );
extern void clc_decrypt_16( clc_bytes_16 * x, clc_key_exp_16 * key );

void clc_encrypt_data_16( unsigned char * in, clc_key_exp_16 * key, long data_len );
void clc_decrypt_data_16( unsigned char * in, clc_key_exp_16 * key, long data_len );

#ifdef __cplusplus
}
#endif

#endif
