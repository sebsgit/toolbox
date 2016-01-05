#ifndef BTREE_H
#define BTREE_H

#ifdef __cplusplus
extern "C" {
#endif

#define btree_set_key(tree, value) (tree)->key = (long)(value)
#define btree_key(tree) (tree)->key

typedef struct btree_t_ {
	void* data;
	long key;
	struct btree_t_* left;
	struct btree_t_* right;
} btree_t;

extern btree_t* btree_new(const long key);
extern btree_t* btree_new_with_data(const long key, const void* data);
extern btree_t* btree_find(const btree_t* root, const long key);
extern int btree_for_each(btree_t* root, int (*callback)(void*));
extern int btree_for_each_with_data(btree_t* root, int (*callback)(void*, void*), void* user_data);
extern btree_t* btree_insert(btree_t* root, const long key, const void* data);
extern btree_t* btree_insert_node(btree_t* root, btree_t* node);
extern btree_t* btree_remove(btree_t* root, const long key);
extern btree_t* btree_remove_node(btree_t* root, btree_t* node);
extern btree_t* btree_remove_with_callback(btree_t* root, const long key, void (*func)(void*));
extern btree_t* btree_remove_node_with_callback(btree_t* root, btree_t* node, void (*func)(void*));
extern btree_t* btree_rebalance(btree_t* root);
extern int btree_depth(btree_t* root);
extern int btree_count(btree_t* root);
extern void btree_free(btree_t* tree);
extern void btree_free_with_callback(btree_t* tree, void (*func)(void*));

#ifdef __cplusplus
}
#endif

#endif
