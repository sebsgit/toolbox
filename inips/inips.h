#ifndef INIPARSE_H
#define INIPARSE_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * .ini file format loader
 * does not care about [group] tags, simply extracts keys and values
 * */

#include "dpstringlist.h"

struct inips_handle_private_t_;

struct inips_handle_t{
    struct inips_handle_private_t_ * d;
};

typedef struct inips_handle_t inips_handle_t;

extern int inips_load_file(const char * path, inips_handle_t * h);
extern int inips_has_key(inips_handle_t * h, const char * key);
extern void inips_get_keys(inips_handle_t * h, dpstringlist_t * keys);
extern void inips_get_stringdp(inips_handle_t * h, const dpstring_t * key, dpstring_t * value);
extern void inips_get_string(inips_handle_t * h, const char * key, dpstring_t * value);
extern const int inips_get_intdp(inips_handle_t * h, const dpstring_t * key);
extern const int inips_get_int(inips_handle_t * h, const char * key);
extern const double inips_get_doubledp(inips_handle_t * h, const dpstring_t * key);
extern const double inips_get_double(inips_handle_t * h, const char * key);
extern void inips_cleanup(inips_handle_t * h);

#ifdef __cplusplus
}
#endif

#endif
