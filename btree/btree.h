#ifndef BTREE_H
#define BTREE_H

#ifdef __cplusplus
extern "C" {
#endif

#define btree_set_key(tree, value) (tree)->key = (long)(value)
#define btree_key(tree) (tree)->key

struct btree_t_ {
	void* data;
	long key;
	struct btree_t_* left;
	struct btree_t_* right;
};
typedef struct btree_t_ btree_t;

extern btree_t* btree_new(const long key);
extern btree_t* btree_find(const btree_t* root, const long key);
extern btree_t* btree_insert(btree_t* root, const long key, const void* data);
extern btree_t* btree_insert_node(btree_t* root, btree_t* node);
extern int btree_depth(btree_t* root);
extern int btree_count(btree_t* root);
extern void btree_free(btree_t* tree);

#ifdef __cplusplus
}
#endif

#endif
