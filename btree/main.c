#include <stdio.h>
#include <assert.h>
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
	const long end_key = 11587;
	long k = start_key;
	for ( ; k<end_key ; ++k)
		root = btree_insert(root, k, 0);
	k = start_key;
	for ( ; k<end_key ; ++k)
		assert(btree_find(root, k));
	printf("elements: %i, depth: %i\n", btree_count(root), btree_depth(root));
	btree_free(root);
	return 0;
}
