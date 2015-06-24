#ifndef _CLC_FILE_H_
#define _CLC_FILE_H_

#ifdef __cplusplus
extern "C" {
#endif

extern void clc_encrypt_file( const char * password, const char * file, const char * file_out );
extern void clc_decrypt_file( const char * password, const char * file, const char * file_out );

#ifdef __cplusplus
}
#endif


#endif
