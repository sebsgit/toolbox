#include "btree.h"
#include <assert.h>
#include <malloc.h>

//TODO cache tree depth
//TODO BFS

static int max(int a, int b) {
	return a > b ? a : b;
}

btree_t* btree_new(const long key) {
	btree_t* result = (btree_t*)malloc(sizeof(btree_t));
	result->left = 0;
	result->right = 0;
	result->key = key;
	return result;
}

void btree_free(btree_t* root) {
	assert(root);
	if (root->left)
		btree_free(root->left);
	if (root->right)
		btree_free(root->right);
	free(root);
}

int btree_depth(btree_t* root) {
	if (root == 0)
		return 0;
	return 1+max(btree_depth(root->left), btree_depth(root->right));
}

int btree_count(btree_t* root) {
	if (root == 0)
		return 0;
	return 1 + btree_count(root->left) + btree_count(root->right);
}

btree_t* btree_find(const btree_t* root, const long key) {
	btree_t* result = 0;
	if (root) {
		if (root->key == key) {
			result = (btree_t*)root;
		} else {
			result = btree_find(root->left, key);
			if (result == 0)
				result = btree_find(root->right, key);
		}
	}
	return result;
}

static btree_t* _btree_insert_node(btree_t* root, btree_t* node) {
	if (root->key < node->key) {
		if (root->right == 0) {
			root->right = node;
		} else {
			_btree_insert_node(root->right, node);
		}
	} else {
		if (root->left == 0) {
			root->left = node;
		} else {
			_btree_insert_node(root->left, node);
		}
	}
	return root;
}

btree_t* btree_insert(btree_t* root, const long key, const void* data) {
	btree_t* node = btree_new(key);
	node->data = (void*)data;
	return btree_insert_node(root, node);
}

static btree_t* _btree_rebalance(btree_t*);

btree_t* btree_insert_node(btree_t* root, btree_t* node) {
	root = _btree_insert_node(root, node);
	root->left = _btree_rebalance(root->left);
	root->right = _btree_rebalance(root->right);
	return _btree_rebalance(root);
}

// replace root with right child
static btree_t* _btree_rotate_left(btree_t* root) {
	if (root->right) {
		btree_t* new_root = root->right;
		btree_t* left_child = root->right->left;
		root->right->left = root;
		root->right = left_child;
		return new_root;
	}
	return root;
}

// replace root with left child
static btree_t* _btree_rotate_right(btree_t* root) {
	if (root->left) {
		btree_t* new_root = root->left;
		btree_t* right_child = root->left->right;
		root->left->right = root;
		root->left = right_child;
		return new_root;
	}
	return root;
}

static btree_t* _btree_rebalance(btree_t* root) {
	if (!root)
		return root;
	const int depth_left = btree_depth(root->left);
	const int depth_right = btree_depth(root->right);
	int rotations = depth_left - depth_right;
	if (rotations < 1 || rotations > 1) {
		if (rotations < 0) {
			rotations = -rotations;
			while (--rotations > 0)
				root = _btree_rotate_left(root);
		} else {
			while (--rotations > 0)
				root = _btree_rotate_right(root);
		}
	}
	return root;
}
