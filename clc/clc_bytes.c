#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "clc_bytes.h"

#ifdef __SSE__
	#include "xmmintrin.h"
#endif

void clc_cpy_b(const clc_bytes_16 * src, clc_bytes_16 * dst){
	memcpy(dst->b, src->b, 16);
}

unsigned char clc_get_b(const clc_bytes_16 * x, short n){
	assert( x && n>-1 && n<16 );
	return x->b[n];
}

void clc_set_b(short n, clc_bytes_16 * x, unsigned char v){
	assert( x && n>-1 && n<16 );
	x->b[n] = v;
}

unsigned char clc_get_b_m(const clc_bytes_16 * x, short rn, short cn){
	assert( x && cn>-1 && cn<4 && rn>-1 && rn<4 );
	return x->b[rn+cn*4];
}

void clc_set_b_m(short rn, short cn, clc_bytes_16 * x, unsigned char v){
	assert( x && cn>-1 && cn<4 && rn>-1 && rn<4 );
	x->b[rn+cn*4] = v;
}

short clc_test_eq( const clc_bytes_16 * x1, const clc_bytes_16 * x2 ){
	return (0 == memcmp(x1->b,x2->b,16));
}

void clc_print_b(const clc_bytes_16 * x){
	short i=0;
	for(i=0 ; i<16 ; ++i){
		printf("%x ",x->b[i]);
	}
	printf("\n");
}

void clc_print_b_mat(const clc_bytes_16 * x){
	short r=0, c=0;
	for(r=0 ; r<4 ; ++r){
		for(c=0 ; c<4 ; ++c){
			printf("%x ",clc_get_b_m(x,r,c));
		}
		printf("\n");
	}
	printf("\n");
}

void clc_xor_16(unsigned char * b1, unsigned char * b2){
	#ifdef __SSE__
		__m128 bytes_sse = _mm_loadu_ps((float *)b1);
		__m128 key_sse = _mm_loadu_ps((float *)b2);
		__m128 xor_sse = _mm_xor_ps(bytes_sse,key_sse);
		_mm_storeu_ps((float *)b1,xor_sse);
	#else
		short i;
		for( i=0 ; i<16 ; ++i ){
			b1[i] ^= b2[i];
		}
	#endif
}

void clc_print_b_20(const clc_bytes_20 * x){
	short i=0;
	for(i=0 ; i<20 ; ++i){
		printf("%x ",x->b[i]);
	}
	printf("\n");
}
