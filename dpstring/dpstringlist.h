#ifndef DPSTRINGLIST_H_
#define DPSTRINGLIST_H_

#ifdef __cplusplus
extern "C"{
#endif

#include "dpstring.h"

/*
 * string list 'class' with O(1) appends, support of iterators and "foreach" operations
 * example use:
 * 		dpstringlist_t list;
 * 		dplist_init(&list);
 * 		
 * 		// append data
 * 		dplist_appends(&list,"a",1);
 * 		dplist_appends(&list,"test",4);
 * 		dplist_appends(&list,"list",4);
 *
 *  	// iterate over data
 * 		dpstring_t value;
 * 		dpstring_init(&value);
 * 		dpstringlist_iterator_t it;
 * 		dplist_beginv(&list, &it, &value); // value == "a"
 * 		dplist_next(&it);
 * 		dplist_value(&it, &value);		// value == "test"
 * 		dplist_nextv(&it, &value);		// value == "list"
 * 
 * 		dplist_is_valid(&it); // returns 0
 * 
 * 		dpstring_cleanup(&value);
 * 		dplist_cleanup(&list);
 * */

struct dpstringlistdata_internal;

struct dpstringlist_t{
	struct dpstringlistdata_internal * d;
	struct dpstringlistdata_internal * last;
};

typedef struct dpstringlist_t dpstringlist_t;

struct dpstringlist_iterator_t{
    struct dpstringlistdata_internal * curr;
};

typedef struct dpstringlist_iterator_t dpstringlist_iterator_t;

extern void dplist_init(dpstringlist_t * list);
extern void dplist_inits(dpstringlist_t * list, const char * buff, const size_t len);
extern void dplist_initdp(dpstringlist_t * list, const dpstring_t * s);
extern void dplist_cleanup(dpstringlist_t * list);

extern int dplist_is_empty(const dpstringlist_t * list);
extern size_t dplist_length(const dpstringlist_t * list);

extern void dplist_appenddp(dpstringlist_t * list, const dpstring_t * s);
extern void dplist_appends(dpstringlist_t * list, const char * s, const size_t len);

extern void dplist_remove_onedp(dpstringlist_t * list, const dpstring_t * s);
extern void dplist_remove_ones(dpstringlist_t * list, const char * s, const size_t len);
extern void dplist_remove_alldp(dpstringlist_t * list, const dpstring_t * s);
extern void dplist_remove_alls(dpstringlist_t * list, const char * s, const size_t len);

extern void dplist_replace_onedp(dpstringlist_t * list, const dpstring_t * x, const dpstring_t * y);
extern void dplist_replace_onedps(dpstringlist_t * list, const dpstring_t * x, const char * y, const size_t len);
extern void dplist_replace_ones(dpstringlist_t * list, const char * x, const size_t lenx, const char * y, const size_t leny);
extern void dplist_replace_alldp(dpstringlist_t * list, const dpstring_t * x, const dpstring_t * y);
extern void dplist_replace_alldps(dpstringlist_t * list, const dpstring_t * x, const char * y, const size_t len);
extern void dplist_replace_alls(dpstringlist_t * list, const char * x, const size_t lenx, const char * y, const size_t leny);

extern void dplist_foreach(const dpstringlist_t * list, void (*func)(const dpstring_t *) );
extern int dplist_containsdp(const dpstringlist_t * list, const dpstring_t * str);
extern int dplist_containss(const dpstringlist_t * list, const char * str, const size_t len);

extern void dplist_join(const dpstringlist_t * list, dpstring_t * out, const char * sep, const size_t seplen);

extern void dplist_print(const dpstringlist_t * list);

extern void dplist_begin(const dpstringlist_t * list, dpstringlist_iterator_t * begin);
extern int dplist_next(dpstringlist_iterator_t * it);
extern int dplist_is_valid(dpstringlist_iterator_t * it);
extern void dplist_value(dpstringlist_iterator_t * it, dpstring_t * out);
/*
 * convenience function, equivalent to call:
 * 		dplist_begin(list,it);
 * 		if (dplist_is_valid(it)){
 * 			dplist_value(it,out);
 * 		}
 * */
extern void dplist_beginv(const dpstringlist_t * list, dpstringlist_iterator_t * it, dpstring_t * out);
/*
 * convenience function, equivalent to call:
 * 		dplist_next(it);
 * 		dplist_value(it,out);
 * */
extern int dplist_nextv(dpstringlist_iterator_t * it, dpstring_t * out);

#ifdef __cplusplus
}
#endif

#endif
