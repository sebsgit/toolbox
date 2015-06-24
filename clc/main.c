#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "clc_encrypt_16.h"
#include "clc_encrypt_24.h"
#include "clc_encrypt_32.h"
#include "clc_md5.h"
#include "clc_sha1.h"
#include "clc_file.h"

void test_16(int num_test){
	int i,j;
	clc_bytes_16 b1, b2;
	clc_key_exp_16 key;
	srand(time(0));
	for(i=0 ; i<num_test ; ++i){
		for(j=0 ; j<16 ; ++j){
			key.b[j] = rand()%256;
			b1.b[j] = rand()%256;
		}
		clc_cpy_b(&b1,&b2);
		clc_expand_key_16(&key);
		clc_encrypt_16(&b1,&key);
		clc_decrypt_16(&b1,&key);
		assert( clc_test_eq(&b1,&b2) );
		printf("\b\b\b\b\b\b\b\b\b\b\b\b\b%i/%i...",i+1,num_test);
	}
}

void test_24(int num_test){
	int i,j;
	clc_bytes_16 b1, b2;
	clc_key_exp_24 key;
	srand(time(0));
	for(i=0 ; i<num_test ; ++i){
		for(j=0 ; j<24 ; ++j){
			key.b[j] = rand()%256;
			if(j<16)
				b1.b[j] = rand()%256;
		}
		clc_cpy_b(&b1,&b2);
		clc_expand_key_24(&key);
		clc_encrypt_24(&b1,&key);
		clc_decrypt_24(&b1,&key);
		assert( clc_test_eq(&b1,&b2) );
		printf("\b\b\b\b\b\b\b\b\b\b\b\b\b%i/%i...",i+1,num_test);
	}
}

void test_32(int num_test){
	int i,j;
	clc_bytes_16 b1, b2;
	clc_key_exp_32 key;
	srand(time(0));
	for(i=0 ; i<num_test ; ++i){
		for(j=0 ; j<32 ; ++j){
			key.b[j] = rand()%256;
			if(j<16)
				b1.b[j] = rand()%256;
		}
		clc_cpy_b(&b1,&b2);
		clc_expand_key_32(&key);
		clc_encrypt_32(&b1,&key);
		clc_decrypt_32(&b1,&key);
		assert( clc_test_eq(&b1,&b2) );
		printf("\b\b\b\b\b\b\b\b\b\b\b\b\b%i/%i...",i+1,num_test);
	}
}

void test(){
	clc_bytes_16 b1;
	clc_key_exp_16 key;
	
	key.b[0] = 0x0f;
	key.b[1] = 0x15;
	key.b[2] = 0x71;
	key.b[3] = 0xc9;
    key.b[4] = 0x47;
    key.b[5] = 0xd9;
    key.b[6] = 0xe8;
    key.b[7] = 0x59;
    key.b[8] = 0x0c;
    key.b[9] = 0xb7;
    key.b[10] = 0xad; 
    key.b[11] = 0xd6;
    key.b[12] = 0xaf;
    key.b[13] = 0x7f;
    key.b[14] = 0x67;
    key.b[15] = 0x98;
    
    memset(b1.b,0x61,16);
	clc_print_b_mat(&b1);
	
	clc_expand_key_16(&key);
	clc_encrypt_16(&b1,&key);
	printf("\n");
	clc_print_b_mat(&b1);
	
	clc_decrypt_16(&b1,&key);
	printf("\n");
	clc_print_b_mat(&b1);
}

int main(int argc, char ** argv){
	const int n = 2000;
	char * test_d = "another text\n but this time its test word to encrypt and decrypt ok";
	char * fout = 0, * fout2 = 0, * fout3 = 0;
	clc_bytes_16 b;
	clc_bytes_20 b20;
	
	if(argc > 1){
		test_d = argv[1];
	}
	if( argc > 4 ){
		fout = argv[2];
		fout2 = argv[3];
		fout3 = argv[4];
	}
	
	test();
	test_16(n);
	test_24(n);
	test_32(n);
	
	clc_md5((const unsigned char*)test_d,strlen(test_d),&b);
	printf("\nmd5: ");
	clc_print_b(&b);
	
	clc_sha1((const unsigned char*)test_d,strlen(test_d),&b20);
	printf("\nsha1 (%lu): ",strlen(test_d));
	clc_print_b_20(&b20);
	
	clc_encrypt_file("test_pass",fout,fout2);
	clc_decrypt_file("test_pass",fout2,fout3);
	return 0;
}

