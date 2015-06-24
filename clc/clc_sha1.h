#ifndef _CLC_SHA_1_H_
#define _CLC_SHA_1_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "clc_bytes.h"
#include <inttypes.h>

struct _clc_sha1_context{
	unsigned char w[64];
	uint32_t tmp[5];
	uint64_t data_len;
	uint_fast8_t buff_len;
};

typedef struct _clc_sha1_context clc_sha1_context;

extern void clc_sha1_initialize( clc_sha1_context * context );
extern void clc_sha1_add_data( const unsigned char * data, long data_len, clc_sha1_context * context );
extern void clc_sha1_finalize( clc_sha1_context * context, clc_bytes_20 * out );

extern void clc_sha1( const unsigned char * data, long data_len, clc_bytes_20 * out );

#ifdef __cplusplus
}
#endif

#endif
