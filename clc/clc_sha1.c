#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include "clc_sha1.h"

#define ROT_L(x, n) ((((x) << (n)) & 0xFFFFFFFF) | ((x) >> (32 - (n))))

unsigned int clc_sha1_init[] = {
	0x67452301,
	0xEFCDAB89,
	0x98BADCFE,
	0x10325476,
	0xC3D2E1F0
};

unsigned char clc_sha1_pad[] = {
	0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

void clc_sha1_initialize( clc_sha1_context * context ){
	memcpy(context->tmp,clc_sha1_init,20);
	context->buff_len = 0;
	context->data_len = 0;
}

void clc_sha1_calc( clc_sha1_context * context ){
	unsigned int x[80];
	unsigned int a,b,c,d,e,f,k;
	unsigned int t;
	int j;
	
	for(t = 0; t < 16; ++t){
		x[t] = ((unsigned) context->w[t * 4]) << 24;
		x[t] |= ((unsigned) context->w[t * 4 + 1]) << 16;
		x[t] |= ((unsigned) context->w[t * 4 + 2]) << 8;
		x[t] |= ((unsigned) context->w[t * 4 + 3]);
	}

	for(j=16 ; j<80 ; ++j){
		t = x[j-3] ^ x[j-8] ^ x[j-14] ^ x[j-16];
		x[j] = ROT_L(t,1);
	}
	
	a = context->tmp[0]; 
	b = context->tmp[1];
	c = context->tmp[2];
	d = context->tmp[3];
	e = context->tmp[4];
	
	for( j=0 ; j<80 ; ++j){
		if( 0 <= j && j<= 19 ){
			f = (b & c) | ((~b) & d);
			k = 0x5A827999;
		} else if( 20 <= j && j <= 39){
			f = b ^ c ^ d;
			k = 0x6ED9EBA1;
		} else if( 40 <= j && j <= 59 ){
			f = (b & c) | (b & d) | (c & d);
			k = 0x8F1BBCDC;
		} else if( 60 <= j && j <= 79){
			f = b ^ c ^ d;
			k = 0xCA62C1D6;
		}

		t = ROT_L(a,5) + f + e + k + x[j];
		e = d;
		d = c;
		c = ROT_L(b,30);
		b = a;
		a = t;
	}
	
	context->tmp[0] += a; 
	context->tmp[1] += b;
	context->tmp[2] += c; 
	context->tmp[3] += d;
	context->tmp[4] += e;
}

void clc_sha1_add_data( const unsigned char * data, long in_data_len, clc_sha1_context * context ){
	long final_len = in_data_len + context->buff_len;
	long i=0;
	context->data_len += in_data_len;
	while( final_len >= 64 ){
		memcpy(context->w+context->buff_len,data+i*64,64-context->buff_len);
		clc_sha1_calc(context);
		final_len -= 64;
		context->buff_len = 0;
		++i;
	}
	memcpy(context->w,data+i*64,final_len);
	context->buff_len = final_len;
}

void clc_sha1_finalize( clc_sha1_context * context, clc_bytes_20 * out ){
	const int64_t length_in_bits = context->data_len*8;
	const int64_t len_bits = (context->data_len*8)%512;
	uint_fast8_t pad_byte, j;
	unsigned char len_buff[8];
	if( len_bits < 448 ){
		pad_byte = (448-len_bits)/8;
	} else if( len_bits > 448 ){
		pad_byte = (448+(512-len_bits))/8;
	} else{
		pad_byte = 64;
	}

	len_buff[ 7 ] = (unsigned char) (length_in_bits >> 0);
	len_buff[ 6 ] = (unsigned char) (length_in_bits >> 8);
	len_buff[ 5 ] = (unsigned char) (length_in_bits >> 16);
	len_buff[ 4 ] = (unsigned char) (length_in_bits >> 24);
	len_buff[ 3 ] = (unsigned char) (length_in_bits >> 32);
	len_buff[ 2 ] = (unsigned char) (length_in_bits >> 40);
	len_buff[ 1 ] = (unsigned char) (length_in_bits >> 48);
	len_buff[ 0 ] = (unsigned char) (length_in_bits >> 56);
	
	if( context->buff_len+pad_byte+8 > 64 ){
		j = 64-context->buff_len;
		memcpy((unsigned char*)context->w+context->buff_len, clc_sha1_pad, j);
		clc_sha1_calc(context);
		memcpy((unsigned char*)context->w, clc_sha1_pad+j, pad_byte-j);
		memcpy((unsigned char*)context->w+(pad_byte-j), len_buff, 8);
	} else{
		memcpy((unsigned char*)context->w+context->buff_len, clc_sha1_pad, pad_byte);
		memcpy((unsigned char*)context->w+context->buff_len+pad_byte, len_buff, 8);
	}
	
	clc_sha1_calc(context);
	
	for(j=0 ; j<5 ; ++j){
		out->b[j*4] = (unsigned char)(context->tmp[j] >> 24);
		out->b[j*4+1] = (unsigned char)(context->tmp[j] >> 16);
		out->b[j*4+2] = (unsigned char)(context->tmp[j] >> 8);
		out->b[j*4+3] = (unsigned char)(context->tmp[j] >> 0);
	}
}

void clc_sha1(const unsigned char * data, long in_data_len, clc_bytes_20 * out){
	clc_sha1_context context;
	clc_sha1_initialize(&context);
	clc_sha1_add_data(data,in_data_len,&context);
	clc_sha1_finalize(&context,out);
}
