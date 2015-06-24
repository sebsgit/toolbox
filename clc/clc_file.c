#include "clc_file.h"
#include "clc_sha1.h"
#include "clc_encrypt_16.h"
#include "clc_encrypt_24.h"
#include "clc_encrypt_32.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

const long max_part_size = 16*1024*1024;

void clc_encrypt_file( const char * password, const char * file, const char * file_out ){
	int ret_code = CLC_ERR_NO_ERR;
	unsigned long num_parts, size, i, part_size;
	unsigned char pad_len, *buff;
	FILE * f = fopen(file,"rb");
	FILE * f_out;
	if(f){
		fseek(f,0,SEEK_END);
		size = ftell(f);
		fseek(f,0,SEEK_SET);
		f_out = fopen(file_out,"wb");
		if(f_out){
			clc_bytes_20 sha1;
			clc_key_exp_16 key;
			pad_len = 16-size%16;
			size += pad_len;
			clc_sha1((const unsigned char*)password,strlen(password),&sha1);
			clc_init_key_16(&key,sha1.b,16);
			clc_expand_key_16(&key);
			
			fwrite(sha1.b,20,1,f_out);
			num_parts = ceil((1.0*size)/max_part_size);
			
			for( i=0 ; i<num_parts ; ++i ){
				part_size = (size > max_part_size ? max_part_size : size);
				buff = (unsigned char*)malloc(part_size);
				fread(buff,part_size,1,f);
				if( i==num_parts-1 ){
					memset(buff+part_size-pad_len,pad_len,pad_len);
				}
				clc_encrypt_data_16(buff,&key,part_size);
				fwrite(buff,part_size,1,f_out);
				size -= part_size;
				free(buff);
			}
			
			fclose(f_out);
		} else{
			ret_code = CLC_ERR_FILE_OPEN_W;
		}
		fclose(f);
	} else{
		ret_code = CLC_ERR_FILE_OPEN_R;
	}
	clc_set_err(ret_code);
}

void clc_decrypt_file(const char * password, const char * file, const char * file_out){
	int ret_code = CLC_ERR_NO_ERR;
	unsigned long num_parts, size, i, part_size;
	unsigned char pad_byte, *buff;
	short sha_ok;
	FILE * f = fopen(file,"rb");
	FILE * f_out;
	if(f){
		f_out = fopen(file_out,"wb");
		fseek(f,0,SEEK_END);
		size = ftell(f)-20;
		fseek(f,0,SEEK_SET);
		
		if(f_out){
			clc_bytes_20 sha1;
			clc_key_exp_16 key;
			clc_sha1((const unsigned char*)password,strlen(password),&sha1);
			buff = (unsigned char*)malloc(20);
			fread(buff,20,1,f);
			sha_ok = memcmp(buff,sha1.b,20);
			free(buff);
			if( sha_ok == 0 ){
				clc_init_key_16(&key,sha1.b,16);
				clc_expand_key_16(&key);
				
				num_parts = ceil((1.0*size)/max_part_size);
				for( i=0 ; i<num_parts-1 ; ++i ){
					part_size = (size > max_part_size ? max_part_size : size);
					buff = (unsigned char*)malloc(part_size);
					fread(buff,part_size,1,f);
					clc_decrypt_data_16(buff,&key,part_size);
					fwrite(buff,part_size,1,f_out);
					size -= part_size;
					free(buff);
				}
				buff = (unsigned char*)malloc(size);
				fread(buff,size,1,f);
				clc_decrypt_data_16(buff,&key,size);
				pad_byte = buff[size-1];
				fwrite(buff,size-pad_byte,1,f_out);
				free(buff);
			} else{
				ret_code = CLC_ERR_WRONG_PASS;
			}
			fclose(f_out);
		} else{
			ret_code = CLC_ERR_FILE_OPEN_W;
		}
		fclose(f);
	} else{
		ret_code = CLC_ERR_FILE_OPEN_R;
	}
	clc_set_err(ret_code);
}
