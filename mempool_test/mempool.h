#ifndef MEM_ALLOC_POOL_H
#define MEM_ALLOC_POOL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <malloc.h>

extern int vpool_init(size_t num_bytes);
extern size_t vpool_bytes_free();
extern size_t vpool_bytes_used();
extern void* vpool_malloc(size_t num_bytes);
//TODO extern void* vpool_realloc(void* ptr, size_t new_size);
extern void vpool_free(void* ptr);
extern void vpool_set_dobule_free_handler(void (*func)(void*));
extern void vpool_cleanup();

#ifdef __cplusplus
}
#endif

#endif
