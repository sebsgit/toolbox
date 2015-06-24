#include <assert.h>
#include <string.h>
#include "clc_encrypt_32.h"

void clc_init_key_32( clc_key_exp_32 * key, const unsigned char * data, int data_len ){
	assert(data_len>0 && data_len<=32 && "FAIL: data length must be [1,32] !");
	clc_init_key(key->b,data,data_len);
	if(data_len<32){
		memset(key->b+data_len, 0, 32-data_len );
	}
}

void clc_expand_key_32( clc_key_exp_32 * key ){
	unsigned char buff[4];
	unsigned char p = 32;
	unsigned short i=1;
	unsigned char a;
	while(p < 240) {
		memcpy(buff, (key->b+p-4), 4);
		if(p % 32 == 0){
			clc_key_sched_core(buff,i);
			i++;
		}
		if(p % 32 == 16) {
			for(a = 0; a < 4; a++) 
				buff[a] = clc_s_box[ buff[a] ];
		}
		for(a = 0; a < 4; a++) {
			key->b[p] = key->b[p - 32] ^ buff[a];
			++p;
		}
	}
}

void clc_encrypt_32( clc_bytes_16 * x, clc_key_exp_32 * key ){
	clc_encrypt(x,key->b,14);
}

void clc_decrypt_32( clc_bytes_16 * x, clc_key_exp_32 * key ){
	clc_decrypt(x,key->b,14,224);
}

void clc_encrypt_data_32( unsigned char * in, clc_key_exp_32 * key, long data_len ){
	const long c = data_len/16;
	long i;
	clc_bytes_16 b;
	clc_expand_key_32(key);
	for(i=0 ; i<c ; ++i){
		memcpy(b.b,in+i*16,16);
		clc_encrypt_32(&b,key);
		memcpy(in+i*16,b.b,16);
	}
}

void clc_decrypt_data_32( unsigned char * in, clc_key_exp_32 * key, long data_len ){
	const long c = data_len/16;
	long i;
	clc_bytes_16 b;
	clc_expand_key_32(key);
	for(i=0 ; i<c ; ++i){
		memcpy(b.b,in+i*16,16);
		clc_decrypt_32(&b,key);
		memcpy(in+i*16,b.b,16);
	}
}
