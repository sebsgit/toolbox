#ifndef _CLC_BYTES_H_
#define _CLC_BYTES_H_

#ifdef __cplusplus
extern "C" {
#endif

struct _bytes{
	unsigned char b[16];
};

struct _bytes_20{
	unsigned char b[20];
};

typedef struct _bytes clc_bytes_16;
typedef struct _bytes_20 clc_bytes_20;

extern unsigned char clc_get_b(const clc_bytes_16 * x, short n);
extern void clc_set_b(short n, clc_bytes_16 * x, unsigned char v);

extern unsigned char clc_get_b_m(const clc_bytes_16 * x, short rn, short cn);
extern void clc_set_b_m(short rn, short cn, clc_bytes_16 * x, unsigned char v);

extern void clc_cpy_b(const clc_bytes_16 * src, clc_bytes_16 * dst);
/*!
 * \brief returns 1 if *x1 == *x2, else return 0
 */
extern short clc_test_eq( const clc_bytes_16 * x1, const clc_bytes_16 * x2 );
extern void clc_print_b(const clc_bytes_16 * x);
extern void clc_print_b_mat(const clc_bytes_16 * x);

extern void clc_xor_16(unsigned char * b1, unsigned char * b2);

extern void clc_print_b_20(const clc_bytes_20 * x);

#ifdef __cplusplus
}
#endif

#endif
