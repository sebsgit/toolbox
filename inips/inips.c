#include "inips.h"
#include "dpfileio.h"
#include <stdio.h>

struct keyval_{
    struct keyval_ * next;
    dpstring_t key;
    dpstring_t value;
};

typedef struct keyval_ keyval_;

struct inips_handle_private_t_{
    keyval_ * keys;
};

typedef struct inips_handle_private_t_ inips_handle_private_t_;

static void priv_append_keys_(keyval_ ** p, dpstring_t * key, dpstring_t * value){
    keyval_ * node = (keyval_*)malloc(sizeof(*node));
    node->next = 0;
    dpstring_init(&node->key);
    dpstring_init(&node->value);
    dpstring_take(&node->key, key);
    dpstring_take(&node->value, value);
    if (! (*p)){
        (*p) = node;
    } else{
        keyval_ * last = *p;
        while (last->next){
            last = last->next;
        }
        last->next = node;
    }
}

static void priv_dealloc_keys_(keyval_ * p){
    while (p){
        keyval_ * next = p->next;
        dpstring_cleanup_fast(&p->key);
        dpstring_cleanup_fast(&p->value);
        free(p);
        p = next;
    }
}
static void priv_dealloc_d_(inips_handle_private_t_ * p){
    if (p){
        priv_dealloc_keys_(p->keys);
        free(p);
    }
}

static void priv_dequote_(dpstring_t * s){
    if (dpstring_starts_withc(s,'"') && dpstring_ends_withc(s,'"')){
        dpstring_t result;
        dpstring_inits(&result,dpstring_toc(s)+1,s->len-2);
        dpstring_take(s,&result);
    }
}

int inips_load_file(const char * path, inips_handle_t * h){
    int result=0;
    dpstringlist_t list;
    dplist_init(&list);
    if ( dpio_split_file(path,"\n",1,&list) ){
        dpstringlist_iterator_t it;
        dpstring_t line;
        h->d = (inips_handle_private_t_*)malloc(sizeof(*h->d));
        h->d->keys = 0;
        dplist_begin(&list,&it);
        while (dplist_is_valid(&it)){
            dpstring_t tmp;
            dpstring_init(&tmp);
            dplist_value(&it,&tmp);
            dpstring_trim(&tmp);
            if ( !dpstring_starts_withc(&tmp,';') && !dpstring_starts_withc(&tmp,'[')){
                dpstringlist_t splitted;
                dplist_init(&splitted);
                dplist_split(&tmp,&splitted,"=",1);
                if (dplist_length(&splitted) == 2){
                    dpstring_t key, value;
                    dpstring_init(&key);
                    dpstring_init(&value);
                    dplist_foreach(&splitted,dpstring_trim);
                    dplist_at(&splitted,0,&key);
                    dplist_at(&splitted,1,&value);
                    priv_dequote_(&value);
                    priv_append_keys_(&h->d->keys,&key,&value);
                }
                dplist_cleanup(&splitted);
            }

            dplist_next(&it);
            dpstring_cleanup_fast(&tmp);
        }
        dpstring_cleanup_fast(&line);
        result = 1;
    }
    dplist_cleanup(&list);
    return result;
}

static keyval_ * locate_node_(keyval_ * start, const char * key){
    keyval_ * node = start;
    while (node){
        if ( dpstring_cmps(&node->key,key,node->key.len) == 0 ){
            return node;
        }
        node = node->next;
    }
    return 0;
}

int inips_has_key(inips_handle_t *h, const char *key){
    return locate_node_(h->d->keys,key) != 0;
}

void inips_get_keys(inips_handle_t * h, dpstringlist_t * keys){
    keyval_ * node = h->d->keys;
    while (node){
        dplist_appenddp(keys,&node->key);
        node = node->next;
    }
}

void inips_get_stringdp(inips_handle_t * h, const dpstring_t * key, dpstring_t * value){
    inips_get_string(h,dpstring_toc(key),value);
}

void inips_get_string(inips_handle_t *h, const char *key, dpstring_t *value){
    keyval_ * n = locate_node_(h->d->keys,key);
    if (n){
        dpstring_copy(value,&n->value);
    }
}

const int inips_get_intdp(inips_handle_t * h, const dpstring_t * key){
    return inips_get_int(h,dpstring_toc(key));
}

const double inips_get_doubledp(inips_handle_t * h, const dpstring_t * key){
    return inips_get_double(h,dpstring_toc(key));
}

const int inips_get_int(inips_handle_t * h, const char * key){
    keyval_ * n = locate_node_(h->d->keys,key);
    if (n){
        return dpstring_toi(&n->value);
    }
    return 0;
}

const double inips_get_double(inips_handle_t * h, const char * key){
    keyval_ * n = locate_node_(h->d->keys,key);
    if (n){
        return dpstring_tod(&n->value);
    }
    return 0.0;
}

void inips_cleanup(inips_handle_t * h){
    if (h){
        priv_dealloc_d_(h->d);
        h->d = 0;
    }
}
