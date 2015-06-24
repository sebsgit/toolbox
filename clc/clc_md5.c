#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include "clc_md5.h"

unsigned char clc_md5_pad[] = {
	0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

unsigned char clc_md5_init[] = {
	0x01, 0x23, 0x45, 0x67,
	0x89, 0xab, 0xcd, 0xef,
	0xfe, 0xdc, 0xba, 0x98,
	0x76, 0x54, 0x32, 0x10
};

/*
long long T(int i){
	const double r = 4294967296*fabs(sin(i));	
	return floor(r);
}
* RCON_i = T(i)*/

#define RCON_1  0xd76aa478
#define RCON_2  0xe8c7b756
#define RCON_3  0x242070db
#define RCON_4  0xc1bdceee
#define RCON_5  0xf57c0faf
#define RCON_6  0x4787c62a
#define RCON_7  0xa8304613 
#define RCON_8  0xfd469501 
#define RCON_9  0x698098d8
#define RCON_10 0x8b44f7af
#define RCON_11 0xffff5bb1
#define RCON_12 0x895cd7be
#define RCON_13 0x6b901122
#define RCON_14 0xfd987193
#define RCON_15 0xa679438e
#define RCON_16 0x49b40821
#define RCON_17 0xf61e2562
#define RCON_18 0xc040b340
#define RCON_19 0x265e5a51
#define RCON_20 0xe9b6c7aa
#define RCON_21 0xd62f105d
#define RCON_22 0x02441453
#define RCON_23 0xd8a1e681
#define RCON_24 0xe7d3fbc8
#define RCON_25 0x21e1cde6
#define RCON_26 0xc33707d6
#define RCON_27 0xf4d50d87
#define RCON_28 0x455a14ed
#define RCON_29 0xa9e3e905
#define RCON_30 0xfcefa3f8
#define RCON_31 0x676f02d9
#define RCON_32 0x8d2a4c8a
#define RCON_33 0xfffa3942
#define RCON_34 0x8771f681
#define RCON_35 0x6d9d6122
#define RCON_36 0xfde5380c
#define RCON_37 0xa4beea44
#define RCON_38 0x4bdecfa9
#define RCON_39 0xf6bb4b60
#define RCON_40 0xbebfbc70
#define RCON_41 0x289b7ec6
#define RCON_42 0xeaa127fa
#define RCON_43 0xd4ef3085
#define RCON_44 0x04881d05
#define RCON_45 0xd9d4d039
#define RCON_46 0xe6db99e5
#define RCON_47 0x1fa27cf8
#define RCON_48 0xc4ac5665
#define RCON_49 0xf4292244
#define RCON_50 0x432aff97
#define RCON_51 0xab9423a7
#define RCON_52 0xfc93a039
#define RCON_53 0x655b59c3
#define RCON_54 0x8f0ccc92
#define RCON_55 0xffeff47d
#define RCON_56 0x85845dd1
#define RCON_57 0x6fa87e4f
#define RCON_58 0xfe2ce6e0
#define RCON_59 0xa3014314
#define RCON_60 0x4e0811a1
#define RCON_61 0xf7537e82
#define RCON_62 0xbd3af235
#define RCON_63 0x2ad7d2bb
#define RCON_64 0xeb86d391

#define ROT_L(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

#define F(x, y, z) (((x) & (y)) | (~(x) & (z)))
#define R1(a, b, c, d, k, s, Ti, x)			\
			t = a + F(b,c,d) + x[k] + Ti;	\
			a = ROT_L(t, s) + b

#define G(x, y, z) (((x) & (z)) | ((y) & ~(z)))
#define R2(a, b, c, d, k, s, Ti, x)			\
			t = a + G(b,c,d) + x[k] + Ti;	\
			a = ROT_L(t, s) + b

#define H(x, y, z) ((x) ^ (y) ^ (z))
#define R3(a, b, c, d, k, s, Ti, x)			\
			t = a + H(b,c,d) + x[k] + Ti;	\
			a = ROT_L(t, s) + b

#define I(x, y, z) ((y) ^ ((x) | ~(z)))
#define R4(a, b, c, d, k, s, Ti, x)			\
			t = a + I(b,c,d) + x[k] + Ti;	\
			a = ROT_L(t, s) + b


void clc_print_buff(unsigned char * x){
	int di;
	for (di = 0; di < 16; ++di)
	    printf("%02x", x[di]);
}

void clc_print_buff_8(unsigned char * x){
	int di;
	for (di = 0; di < 8; ++di)
	    printf("%02x", x[di]);
}

void clc_md5_initialize( clc_md5_context * context ){
	memcpy(context->md5,clc_md5_init,16);
	context->buff_len = 0;
	context->data_len = 0;
}

void clc_md5_calc( clc_md5_context * context ){
	unsigned int a,b,c,d,t;
	a = context->md5[0];
	b = context->md5[1];
	c = context->md5[2];
	d = context->md5[3];

	R1(a, b, c, d,  0,  7,  RCON_1, context->buff);
	R1(d, a, b, c,  1, 12,  RCON_2, context->buff);
	R1(c, d, a, b,  2, 17,  RCON_3, context->buff);
	R1(b, c, d, a,  3, 22,  RCON_4, context->buff);
	R1(a, b, c, d,  4,  7,  RCON_5, context->buff);
	R1(d, a, b, c,  5, 12,  RCON_6, context->buff);
	R1(c, d, a, b,  6, 17,  RCON_7, context->buff);
	R1(b, c, d, a,  7, 22,  RCON_8, context->buff);
	R1(a, b, c, d,  8,  7,  RCON_9, context->buff);
	R1(d, a, b, c,  9, 12, RCON_10, context->buff);
	R1(c, d, a, b, 10, 17, RCON_11, context->buff);
	R1(b, c, d, a, 11, 22, RCON_12, context->buff);
	R1(a, b, c, d, 12,  7, RCON_13, context->buff);
	R1(d, a, b, c, 13, 12, RCON_14, context->buff);
	R1(c, d, a, b, 14, 17, RCON_15, context->buff);
	R1(b, c, d, a, 15, 22, RCON_16, context->buff);
	
	R2(a, b, c, d,  1,  5, RCON_17, context->buff);
	R2(d, a, b, c,  6,  9, RCON_18, context->buff);
	R2(c, d, a, b, 11, 14, RCON_19, context->buff);
	R2(b, c, d, a,  0, 20, RCON_20, context->buff);
	R2(a, b, c, d,  5,  5, RCON_21, context->buff);
	R2(d, a, b, c, 10,  9, RCON_22, context->buff);
	R2(c, d, a, b, 15, 14, RCON_23, context->buff);
	R2(b, c, d, a,  4, 20, RCON_24, context->buff);
	R2(a, b, c, d,  9,  5, RCON_25, context->buff);
	R2(d, a, b, c, 14,  9, RCON_26, context->buff);
	R2(c, d, a, b,  3, 14, RCON_27, context->buff);
	R2(b, c, d, a,  8, 20, RCON_28, context->buff);
	R2(a, b, c, d, 13,  5, RCON_29, context->buff);
	R2(d, a, b, c,  2,  9, RCON_30, context->buff);
	R2(c, d, a, b,  7, 14, RCON_31, context->buff);
	R2(b, c, d, a, 12, 20, RCON_32, context->buff);

	R3(a, b, c, d,  5,  4, RCON_33, context->buff);
	R3(d, a, b, c,  8, 11, RCON_34, context->buff);
	R3(c, d, a, b, 11, 16, RCON_35, context->buff);
	R3(b, c, d, a, 14, 23, RCON_36, context->buff);
	R3(a, b, c, d,  1,  4, RCON_37, context->buff);
	R3(d, a, b, c,  4, 11, RCON_38, context->buff);
	R3(c, d, a, b,  7, 16, RCON_39, context->buff);
	R3(b, c, d, a, 10, 23, RCON_40, context->buff);
	R3(a, b, c, d, 13,  4, RCON_41, context->buff);
	R3(d, a, b, c,  0, 11, RCON_42, context->buff);
	R3(c, d, a, b,  3, 16, RCON_43, context->buff);
	R3(b, c, d, a,  6, 23, RCON_44, context->buff);
	R3(a, b, c, d,  9,  4, RCON_45, context->buff);
	R3(d, a, b, c, 12, 11, RCON_46, context->buff);
	R3(c, d, a, b, 15, 16, RCON_47, context->buff);
	R3(b, c, d, a,  2, 23, RCON_48, context->buff);

	R4(a, b, c, d,  0,  6, RCON_49, context->buff);
	R4(d, a, b, c,  7, 10, RCON_50, context->buff);
	R4(c, d, a, b, 14, 15, RCON_51, context->buff);
	R4(b, c, d, a,  5, 21, RCON_52, context->buff);
	R4(a, b, c, d, 12,  6, RCON_53, context->buff);
	R4(d, a, b, c,  3, 10, RCON_54, context->buff);
	R4(c, d, a, b, 10, 15, RCON_55, context->buff);
	R4(b, c, d, a,  1, 21, RCON_56, context->buff);
	R4(a, b, c, d,  8,  6, RCON_57, context->buff);
	R4(d, a, b, c, 15, 10, RCON_58, context->buff);
	R4(c, d, a, b,  6, 15, RCON_59, context->buff);
	R4(b, c, d, a, 13, 21, RCON_60, context->buff);
	R4(a, b, c, d,  4,  6, RCON_61, context->buff);
	R4(d, a, b, c, 11, 10, RCON_62, context->buff);
	R4(c, d, a, b,  2, 15, RCON_63, context->buff);
	R4(b, c, d, a,  9, 21, RCON_64, context->buff);
	
	context->md5[0] += a; 
	context->md5[1] += b;
	context->md5[2] += c; 
	context->md5[3] += d;
}

void clc_md5_add_data( const unsigned char * data, long in_data_len, clc_md5_context * context ){
	long final_len = in_data_len + context->buff_len;
	long i=0;
	context->data_len += in_data_len;
	while( final_len >= 64 ){
		memcpy(context->buff+context->buff_len,data+i*64,64-context->buff_len);
		clc_md5_calc(context);
		final_len -= 64;
		context->buff_len = 0;
		++i;
	}
	memcpy(context->buff,data+i*64,final_len);
	context->buff_len = final_len;
}

void clc_md5_finalize( clc_md5_context * context, clc_bytes_16 * out ){
	const int64_t length_in_bits = context->data_len*8;
	const int64_t len_bits = (context->data_len*8)%512;
	uint_fast8_t tmp, pad_byte;
	unsigned char len_buff[8];
	if( len_bits < 448 ){
		pad_byte = (448-len_bits)/8;
	} else if( len_bits > 448 ){
		pad_byte = (448+(512-len_bits))/8;
	} else{
		pad_byte = 64;
	}

	len_buff[ 0 ] = (unsigned char) (length_in_bits >> 0);
	len_buff[ 1 ] = (unsigned char) (length_in_bits >> 8);
	len_buff[ 2 ] = (unsigned char) (length_in_bits >> 16);
	len_buff[ 3 ] = (unsigned char) (length_in_bits >> 24);
	len_buff[ 4 ] = (unsigned char) (length_in_bits >> 32);
	len_buff[ 5 ] = (unsigned char) (length_in_bits >> 40);
	len_buff[ 6 ] = (unsigned char) (length_in_bits >> 48);
	len_buff[ 7 ] = (unsigned char) (length_in_bits >> 56);
	
	if( context->buff_len+pad_byte+8 > 64 ){
		tmp = 64-context->buff_len;
		memcpy((unsigned char*)context->buff+context->buff_len, clc_md5_pad, tmp);
		clc_md5_calc(context);
		memcpy((unsigned char*)context->buff, clc_md5_pad+tmp, pad_byte-tmp);
		memcpy((unsigned char*)context->buff+(pad_byte-tmp), len_buff, 8);
	} else{
		memcpy((unsigned char*)context->buff+context->buff_len, clc_md5_pad, pad_byte);
		memcpy((unsigned char*)context->buff+context->buff_len+pad_byte, len_buff, 8);
	}
	
	clc_md5_calc(context);
	
	memcpy(out->b,context->md5,16);
}

void clc_md5( const unsigned char * data, long data_len, clc_bytes_16 * out ){
	clc_md5_context c;
	clc_md5_initialize(&c);
	clc_md5_add_data(data,data_len,&c);
	clc_md5_finalize(&c,out);
}
