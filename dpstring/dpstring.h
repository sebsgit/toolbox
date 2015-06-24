#ifndef DPSTRING_H_
#define DPSTRING_H_

#ifdef __cplusplus
extern "C"{
#endif

#include <inttypes.h>
#include <stdlib.h>

/*
 * string 'class' with short string optimization
 * strings up to 16 characters are kept on the stack
 * longer string will have memory allocated from the heap
 * 
 * as the string expands, memory will be allocated in batches, in order
 * to avoid frequent allocations
 * 
 * do not use fields in this structure directly, use the "member functions" instead
 * 
 * example use:
 * 
 * 		dpstring_t string;
 * 		dpstring_inits(&string, "hello", 5);
 * 		dpstring_print(&string);
 * 		dpstring_cleanup(&string);
 * 
 * note about heap allocations: 
 * 	if you allocate the dpstring_t structure on the heap, then you are responsible to deallocate it
 *  dpstring_cleanup* functions do not free the parameter pointer (due to possible stack allocation)
 * */
struct dpstring_t{
	uint32_t len;
	uint32_t buffsize;
	char data[16];
	char * ptr;
};

typedef struct dpstring_t dpstring_t;

extern void dpstring_init(dpstring_t * str);
extern void dpstring_inits(dpstring_t * str, const char * buff, const size_t len);
extern void dpstring_initcc(dpstring_t * str, const char c, const size_t count);
/*
 * cleans up the string, free any allocated resources and zero buffers and fields
 * this function does not "free" the str pointer itself
 * */
extern void dpstring_cleanup(dpstring_t * str);
/*
 * cleans up any resources allocated to str pointer, but leaves everything else untouched
 * care should be taken in order to not use the str after calling this function, due to
 * possible dangling pointers
 * 
 * this function does not attempt to "free" the str pointer
 * */
extern void dpstring_cleanup_fast(dpstring_t * str);

extern int dpstring_is_empty(const dpstring_t * str);
extern const uint32_t dpstring_length(const dpstring_t * str);
extern const char dpstring_getc(const dpstring_t * str, const size_t pos);

/*
 * copy src to dest
 * */
extern void dpstring_copy(dpstring_t * dest, const dpstring_t * src);
extern void dpstring_copys(dpstring_t * dest, const char * buff, const size_t len);
/*
 * perform "move" operation from "to_emtpy" into "str"
 * this will empty the string "to_empty"
 * this function is meant to be used as "fast copy", when you dont care about the second parameter
 */
extern void dpstring_take(dpstring_t * str, dpstring_t * to_empty);

/*
 * store "str" in char buffer
 * */
extern void dpstring_store(char * buff, const dpstring_t * str, const size_t len);

/*
 * prints "str" to stdout
 * */
extern void dpstring_print(const dpstring_t * str);

extern void dpstring_appendc(dpstring_t * str, char c);
extern void dpstring_appenddp(dpstring_t * str, const dpstring_t * buff);
extern void dpstring_appends(dpstring_t * str, const char * buff, const size_t len);

extern int dpstring_cmp(const dpstring_t * str1, const dpstring_t * str2);
extern int dpstring_cmps(const dpstring_t * str1, const char * str2, const size_t len);

extern int dpstring_locatec(const dpstring_t * str, const size_t pos, const char c);
extern int dpstring_locates(const dpstring_t * str, const size_t pos, const char * buff, const size_t len);
extern int dpstring_locatedp(const dpstring_t * str, const size_t pos, const dpstring_t * buff);
extern void dpstring_replacecc(dpstring_t * str, const char x, const char y);
extern void dpstring_replacecs(dpstring_t * str, const char x, const char * buff, const size_t len);
extern void dpstring_replacess(dpstring_t * str, const char * x, const size_t lenx, const char * buff, const size_t lenbuff);
extern void dpstring_replacedp(dpstring_t * str, const dpstring_t * x, const dpstring_t * y);
extern void dpstring_replacedpi(dpstring_t * str, const dpstring_t * x, const dpstring_t * y, const size_t start_pos);
extern void dpstring_replacei(dpstring_t * str, const size_t a, const size_t b, const dpstring_t * y);

extern void dpstring_boundc(const dpstring_t * str, const char a, const char b, const size_t start, dpstring_t * out);
/*
 * removes white space characters from beginning and end of string
 * whitespace characters include: '\t', '\n', '\v', '\f', '\r', and ' '
 * */
extern void dpstring_trim(dpstring_t * str);

/*
 * convert to plain c string pointer
 * this function does not allocate any resources
 * you don't have to free the returned pointer
 * */
extern const char * dpstring_toc(const dpstring_t * str);

#ifdef __cplusplus
}
#endif

#endif
