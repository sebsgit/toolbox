#include "mempool.h"
#include <assert.h>

// TODO
// tree / hash of available allocations instead of list
// - [size1, <list of blocks == size1>]
// - [size2, <list of blocks == size2>]
// ..
//TODO initialize with some memory for bookkeeping

typedef void (*_vpool_double_free_callback)(void*);
_vpool_double_free_callback _vpool_double_free_handler = 0;

typedef struct {
	void* data;
	int used;
} allocation_block_t;
static void _vpool_init_alloc_block(allocation_block_t* block) {
	block->used = 0;
	block->data = 0;
}
static void _vpool_free_alloc_block(allocation_block_t* block) {
	if (block->used == 0)
		free(block->data);
}

typedef struct {
	allocation_block_t* allocations;
	int count;
} allocations_data_t;
static void _vpool_init_allocations(allocations_data_t* data){
	data->allocations = 0;
	data->count = 0;
}
static void _vpool_free_allocations(allocations_data_t* data){
	int i=0;
	for ( ; i<data->count ; ++i)
		_vpool_free_alloc_block(&data->allocations[i]);
	free(data->allocations);
}
static void _vpool_add_allocation(allocations_data_t* data, void* ptr){
	++data->count;
	data->allocations = realloc(data->allocations, data->count * sizeof(allocation_block_t));
	allocation_block_t block;
	_vpool_init_alloc_block(&block);
	block.data = ptr;
	block.used = 1;
	data->allocations[data->count - 1] = block;
}
static void* _vpool_get_allocation_block(allocations_data_t* data) {
	void* result = 0;
	if (data->count > 0) {
		int i=0;
		for ( ; i<data->count ; ++i) {
			if (data->allocations[i].used == 0) {
				data->allocations[i].used = 1;
				result = data->allocations[i].data;
				break;
			}
		}
	}
	return result;
}
static int _vpool_return_block(allocations_data_t* data, void* block) {
	assert(data);
	if (data->count) {
		int i=0;
		for ( ; i<data->count ; ++i)
			if (data->allocations[i].data == block) {
				if (data->allocations[i].used == 0 && _vpool_double_free_handler)
					_vpool_double_free_handler(block);
				data->allocations[i].used = 0;
				return 1;
			}
	}
	return 0;
}

typedef struct {
	size_t alloc_size;
	allocations_data_t allocs;
} node_info_t;
static void _vpool_init_node(node_info_t* node) {
	node->alloc_size = 0;
	_vpool_init_allocations(&node->allocs);
}
static void _vpool_cleanup_node(node_info_t* node) {
	assert(node);
	_vpool_free_allocations(&node->allocs);
}
static void _vpool_add_to_node(node_info_t* node, void* block) {
	assert(node);
	_vpool_add_allocation(&node->allocs, block);
}
static void* _vpool_get_from_node(node_info_t* node) {
	assert(node);
	return _vpool_get_allocation_block(&node->allocs);
}
static size_t _vpool_return_block_to_node(node_info_t* info, void* data) {
	assert(info);
	return _vpool_return_block(&info->allocs, data) ? info->alloc_size : 0;
}

typedef struct {
	node_info_t** info;
	int count;
	size_t size_left;
} vpool_info_t;
static void _vpool_init_pool(vpool_info_t* pool, size_t num_bytes) {
	pool->info = 0;
	pool->count = 0;
	pool->size_left = num_bytes;
}
static void _vpool_cleanup_pool(vpool_info_t* pool){
	if (pool->info) {
		int i=0;
		for ( ; i<pool->count ; ++i){
			_vpool_cleanup_node(pool->info[i]);
			free(pool->info[i]);
		}
		free(pool->info);
		pool->info = 0;
	}
	pool->count = 0;
	pool->size_left = 0;
}
static node_info_t* _vpool_find_node(vpool_info_t* pool, size_t num_bytes) {
	assert(pool);
	if (pool->count) {
		int i=0;
		for ( ; i<pool->count ; ++i) {
			assert(pool->info);
			if (pool->info[i]->alloc_size == num_bytes)
				return pool->info[i];
		}
	}
	return 0;
}
static node_info_t* _vpool_add_node(vpool_info_t* pool, size_t num_bytes) {
	node_info_t* result = (node_info_t*)malloc(sizeof(node_info_t));
	_vpool_init_node(result);
	result->alloc_size = num_bytes;
	if (!pool->info) {
		pool->info = (node_info_t**)malloc(sizeof(node_info_t*));
		pool->count = 1;
	} else {
		++pool->count;
		pool->info = (node_info_t**)realloc(pool->info, pool->count * sizeof(node_info_t*));
	}
	pool->info[pool->count - 1] = result;
	return result;
}

static void* _vpool_get_allocation(vpool_info_t* pool, size_t num_bytes) {
	node_info_t* info = _vpool_find_node(pool, num_bytes);
	if (info) {
		void* block = _vpool_get_from_node(info);
		if (block)
			return block;
	}
	return 0;
}

static void _vpool_insert_allocation(vpool_info_t* pool, void* block, size_t num_bytes) {
	if (num_bytes < pool->size_left) {
		node_info_t* node = _vpool_find_node(pool, num_bytes);
		if (!node)
			node = _vpool_add_node(pool, num_bytes);
		_vpool_add_to_node(node, block);
		pool->size_left -= num_bytes;
	}
}

static size_t _vpool_return_to_pool(vpool_info_t* pool, void* block) {
	assert(pool);
	assert(block);
	if (pool->count) {
		int i=0;
		for ( ; i<pool->count ; ++i) {
			size_t size = _vpool_return_block_to_node(pool->info[i], block);
			if (size > 0)
				return size;
		}
	}
	return 0;
}

static vpool_info_t vpool;

int vpool_init(size_t num_bytes) {
	assert(num_bytes > 0);
	_vpool_init_pool(&vpool, num_bytes);
	return 0;
}

void* vpool_malloc(size_t num_bytes) {
	void* block = _vpool_get_allocation(&vpool, num_bytes);
	if (block)
		return block;
	block = malloc(num_bytes);
	_vpool_insert_allocation(&vpool, block, num_bytes);
	return block;
}

void* vpool_realloc(void* ptr, size_t new_size) {
	return realloc(ptr, new_size);
}

void vpool_free(void* ptr) {
	if (_vpool_return_to_pool(&vpool, ptr) == 0)
		free(ptr);
}

void vpool_cleanup() {
	_vpool_cleanup_pool(&vpool);
}

void vpool_set_dobule_free_handler(void (*func)(void*)) {
	_vpool_double_free_handler = func;
}
