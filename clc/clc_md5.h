#ifndef _CLC_M_D_5_H_
#define _CLC_M_D_5_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "clc_bytes.h"
#include <inttypes.h>

struct _clc_md5_context{
	int32_t buff[16];
	uint32_t md5[4];
	uint64_t data_len;
	uint_fast8_t buff_len;
};

typedef struct _clc_md5_context clc_md5_context;

extern void clc_md5_initialize( clc_md5_context * context );
extern void clc_md5_add_data( const unsigned char * data, long data_len, clc_md5_context * context );
extern void clc_md5_finalize( clc_md5_context * context, clc_bytes_16 * out );

extern void clc_md5( const unsigned char * data, long data_len, clc_bytes_16 * out );

#ifdef __cplusplus
}
#endif

#endif
