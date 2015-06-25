#ifndef DPLOADER_H
#define DPLOADER_H

/*
 * shared library loader
 * */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * load shared library from "path", lazy initialization is used
 * on success returns non-null handle that should be closed with dp_close
 * on failure 0 is returned
 * */
extern void * dp_load(const char * path);
/*
 * attempt to load symbol "sym" from library handle
 * on failure 0 is returned
 * on success returns address of the symbol in shared library
 * */
extern void * dp_symbol(void * h, const char * sym);
/*
 * returns human-readable string describing last error
 * */
extern const char * dp_error();
/*
 * close library handle
 * */
extern void dp_close(void * h);

#ifdef __cplusplus
}
#endif

#endif
