#include <assert.h>
#include <string.h>
#include "clc_encrypt_16.h"

void clc_init_key_16( clc_key_exp_16 * key, const unsigned char * data, int data_len ){
	assert(data_len>0 && data_len<=16 && "FAIL: data length must be [1,16] !");
	clc_init_key(key->b,data,data_len);
	if(data_len<16){
		memset(key->b+data_len, 0, 16-data_len );
	}
}

void clc_expand_key_16( clc_key_exp_16 * key ){
	unsigned char buff[4];
	unsigned char p = 16;
	unsigned short i=1;
	unsigned char a;
	while(p < 176) {
		memcpy(buff, (key->b+p-4), 4);
		if(p % 16 == 0){
			clc_key_sched_core(buff,i);
			i++;
		}
		for(a = 0; a < 4; a++) {
			key->b[p] = key->b[p - 16] ^ buff[a];
			++p;
		}
	}
}

void clc_encrypt_16( clc_bytes_16 * x, clc_key_exp_16 * key ){
	clc_encrypt(x,key->b,10);
}

void clc_decrypt_16( clc_bytes_16 * x, clc_key_exp_16 * key ){
	clc_decrypt(x,key->b,10,160);
}

void clc_encrypt_data_16( unsigned char * in, clc_key_exp_16 * key, long data_len ){
	const long c = data_len/16;
	long i;
	clc_bytes_16 b;
	clc_expand_key_16(key);
	for(i=0 ; i<c ; ++i){
		memcpy(b.b,in+i*16,16);
		clc_encrypt_16(&b,key);
		memcpy(in+i*16,b.b,16);
	}
}

void clc_decrypt_data_16( unsigned char * in, clc_key_exp_16 * key, long data_len ){
	const long c = data_len/16;
	long i;
	clc_bytes_16 b;
	clc_expand_key_16(key);
	for(i=0 ; i<c ; ++i){
		memcpy(b.b,in+i*16,16);
		clc_decrypt_16(&b,key);
		memcpy(in+i*16,b.b,16);
	}
}
