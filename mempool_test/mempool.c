#include "mempool.h"
#include "btree.h"
#include <assert.h>

//TODO remove global variables
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
    btree_t* alloc_data;
	size_t size_left;
    size_t size_used;
} vpool_info_t;
static void _vpool_init_pool(vpool_info_t* pool, size_t num_bytes) {
    pool->alloc_data = 0;
	pool->size_left = num_bytes;
    pool->size_used = 0;
}
static void _vpool_node_cleanup_callback(void* raw) {
    if (raw)
        _vpool_cleanup_node((node_info_t*)raw);
}
static void _vpool_cleanup_pool(vpool_info_t* pool){
    if (pool->alloc_data) {
        btree_free_with_callback(pool->alloc_data, _vpool_node_cleanup_callback);
        pool->alloc_data = 0;
	}
	pool->size_left = 0;
    pool->size_used = 0;
}
static node_info_t* _vpool_find_node(vpool_info_t* pool, size_t num_bytes) {
	assert(pool);
    if (pool->alloc_data) {
        btree_t* node = btree_find(pool->alloc_data, num_bytes);
        if (node)
            return (node_info_t*)node->data;
	}
	return 0;
}
static node_info_t* _vpool_add_node(vpool_info_t* pool, size_t num_bytes) {
	node_info_t* result = (node_info_t*)malloc(sizeof(node_info_t));
	_vpool_init_node(result);
	result->alloc_size = num_bytes;
    if (!pool->alloc_data) {
        pool->alloc_data = btree_new_with_data(num_bytes, result);
	} else {
        pool->alloc_data = btree_insert(pool->alloc_data, num_bytes, result);
	}
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
        pool->size_used += num_bytes;
	}
}

//TODO change this after support for iteration in tree

static size_t _last_result = 0;
static void* _last_block = 0;

static void _vpool_for_each(btree_t* root, void (*callback)(void*)) {
    if (root) {
        if (_last_result == 0)
            callback(root->data);
        if (_last_result == 0)
            _vpool_for_each(root->left, callback);
        if (_last_result == 0)
            _vpool_for_each(root->right, callback);
    }
}
static void _vpool_node_search(void* raw) {
    if (raw) {
        node_info_t* node = (node_info_t*)raw;
        _last_result = _vpool_return_block_to_node(node, _last_block);
    }
}
static size_t _vpool_return_to_pool(vpool_info_t* pool, void* block) {
	assert(pool);
	assert(block);
    _last_result = 0;
    if (pool->alloc_data) {
        _last_block = block;
        _vpool_for_each(pool->alloc_data, _vpool_node_search);
	}
    return _last_result;
}

static vpool_info_t vpool;

int vpool_init(size_t num_bytes) {
	assert(num_bytes > 0);
	_vpool_init_pool(&vpool, num_bytes);
	return 0;
}

size_t vpool_bytes_free() {
    return vpool.size_left;
}

size_t vpool_bytes_used() {
    return vpool.size_used;
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
