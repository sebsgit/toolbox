#ifndef _CLC_ENCRYPT_BASE_H_
#define _CLC_ENCRYPT_BASE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "clc_bytes.h"
#include "clc_error.h"

extern unsigned char clc_s_box[256];
extern unsigned char clc_s_box_inv[256];
extern unsigned char clc_E_tab[256];
extern unsigned char clc_L_tab[256];
extern unsigned char clc_Rcon_tab[16];

struct _key_exp_16{
	unsigned char b[176];
};
struct _key_exp_24{
	unsigned char b[208];
};
struct _key_exp_32{
	unsigned char b[240];
};

typedef struct _key_exp_16 clc_key_exp_16;
typedef struct _key_exp_24 clc_key_exp_24;
typedef struct _key_exp_32 clc_key_exp_32;

extern void clc_print_key_16( const clc_key_exp_16 * key );
extern void clc_print_key_24( const clc_key_exp_24 * key );
extern void clc_print_key_32( const clc_key_exp_32 * key );

extern void clc_sub_bytes( clc_bytes_16 * x );
extern void clc_sub_bytes_rev( clc_bytes_16 * x );
extern void clc_shift_row_right(short rn, clc_bytes_16 * x);
extern void clc_shift_row_left(short rn, clc_bytes_16 * x);
extern void clc_shift_row_left_n(short rn, clc_bytes_16 * x, short c);
extern void clc_shift_row_right_n(short rn, clc_bytes_16 * x, short c);
extern void clc_shift_row(clc_bytes_16 * x);
extern void clc_shift_row_rev( clc_bytes_16 * x );
extern unsigned char clc_mult_L(unsigned char b1, unsigned char b2);
extern void clc_mix_column(short cn, clc_bytes_16 * x);
extern void clc_mix_column_rev(short cn, clc_bytes_16 * x);
extern void clc_mix_columns( clc_bytes_16 * x );
extern void clc_mix_columns_rev( clc_bytes_16 * x );

extern void clc_rot_word(unsigned char * bp);
extern void clc_sub_word( unsigned char * bp );
extern void clc_key_sched_core( unsigned char * b_in, short i );

extern void clc_init_key( unsigned char * key, const unsigned char * data, int data_len );
extern void clc_add_round_key( clc_bytes_16 * x, unsigned char * key_b, short n_round );
extern void clc_add_round_key_rev( clc_bytes_16 * x, unsigned char * key_b, short n_round, int key_size );
extern void clc_encrypt( clc_bytes_16 * x, unsigned char * key, short n_rounds );
extern void clc_decrypt( clc_bytes_16 * x, unsigned char * key, short n_rounds, int key_size );

extern short clc_rounds( short key_len );

extern long clc_fill_to_16( const unsigned char * src, unsigned char ** dest, long len );

#ifdef __cplusplus
}
#endif

#endif
