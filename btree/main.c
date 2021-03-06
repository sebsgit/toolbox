#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include "btree.h"

int main(int argc, char ** argv){
	btree_t* root = btree_new(17);
	root = btree_insert(root, 23, 0);
	root = btree_insert(root, 1, 0);
	assert(root->key == 17);
	root = btree_insert(root, 2, 0);
	root = btree_insert(root, 3, 0);
	root = btree_insert(root, 19, 0);
	assert(root->key == 17);
	assert(btree_depth(root) == 3);
	const long start_key = 150;
	const long end_key = 1587;
	long k = start_key;
	for ( ; k<end_key ; ++k)
		root = btree_insert(root, k, 0);
	k = start_key;
	for ( ; k<end_key ; ++k)
		assert(btree_find(root, k));
	int count = btree_count(root);
	printf("elements: %i, depth: %i\n", count, btree_depth(root));
	root = btree_remove(root, start_key);
	root = btree_remove(root, (start_key + end_key) / 2);
	root = btree_remove(root, end_key - 1);
	assert(btree_count(root) == count - 3);
	btree_free(root);
	
	root = btree_new_with_data(23, malloc(123));
	root = btree_insert(root, 34, malloc(643));
	root = btree_insert(root, 3, malloc(852));
	root = btree_insert(root, -5, malloc(63));
	root = btree_remove_with_callback(root, 34, free);
	btree_free_with_callback(root, free);
	return 0;
}
