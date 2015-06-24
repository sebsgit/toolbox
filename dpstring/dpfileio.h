#ifndef DPIFLEIO_H
#define DPIFLEIO_H

#ifdef __cplusplus
extern "C"{
#endif

#include "dpstring.h"
#include "dpstringlist.h"

/*
 * attempt to load entire file to string
 * */
extern int dpio_load_file(const char * path, dpstring_t * out);
/*
 * splits the file content to string list using given separator
 * */
extern int dpio_split_file(const char * path, const char * sep, const size_t seplen, dpstringlist_t * out);

extern void dpio_writes(const char * path, const dpstring_t * to_write);
extern void dpio_appends(const char * path, const dpstring_t * to_append);

#ifdef __cplusplus
}
#endif

#endif
