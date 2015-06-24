#include "dpstringlist.h"
#include <stdio.h>

struct dpstringlistdata_internal{
	struct dpstringlistdata_internal * next;
	struct dpstringlistdata_internal * prev;
	dpstring_t value;
};

typedef struct dpstringlistdata_internal list_data_;

/* ITERATORS */

void dplist_begin(const dpstringlist_t * list, dpstringlist_iterator_t * begin){
    begin->curr = list->d;
}

int dplist_next(dpstringlist_iterator_t * it){
    it->curr = it->curr->next;
    return it->curr != 0;
}

int dplist_is_valid(dpstringlist_iterator_t *it){
    return it->curr != 0;
}

void dplist_value(dpstringlist_iterator_t * it, dpstring_t *out){
    dpstring_cleanup(out);
    dpstring_inits(out, dpstring_toc(&it->curr->value), it->curr->value.len);
}


void dplist_beginv(const dpstringlist_t * list, dpstringlist_iterator_t * it, dpstring_t * out){
	dplist_begin(list,it);
	if (dplist_is_valid(it)){
		dplist_value(it,out);
	}
}

int dplist_nextv(dpstringlist_iterator_t * it, dpstring_t * out){
	const int n = dplist_next(it);
	if (n){
		dplist_value(it,out);
	}
	return n;
}

/* STRING LIST */

static void detach_and_delete_(dpstringlist_t * list, list_data_ * node){
	if (node){
		if (node == list->last)
			list->last = node->prev;
		if (node->prev){
			node->prev->next = node->next;
		}
		if (node->next){
			node->next->prev = node->prev;
		}
		dpstring_cleanup_fast(&node->value);
		if (node == list->d)
			list->d = 0;
		free(node);
	}
}

static list_data_ * get_node_internals_(list_data_ * d, const char * buff, const size_t len){
	while (d){
		if (dpstring_cmps(&d->value,buff,len)==0)
			return d;
		d = d->next;
	}
	return 0;
}


void dplist_init(dpstringlist_t * list){
	list->d = 0;
	list->last = 0;
}

void dplist_inits(dpstringlist_t * list, const char * buff, const size_t len){
	list->d = (list_data_*)malloc(sizeof(*list->d));
	list->d->prev = 0;
	list->d->next = 0;
	list->last = list->d;
	dpstring_inits(&list->d->value,buff,len);
}

void dplist_initdp(dpstringlist_t * list, const dpstring_t * s){
	dplist_inits(list,dpstring_toc(s),s->len);
}

void dplist_cleanup(dpstringlist_t * list){
	list_data_ * p = list->d;
	list_data_ * next = 0;
	while (p) {
		dpstring_cleanup_fast(&p->value);
		next = p->next;
		free(p);
		p = next;
	}
	list->d = 0;
	list->last = 0;
}

int dplist_is_empty(const dpstringlist_t *list){
    return list->d == 0;
}

size_t dplist_length(const dpstringlist_t * list){
	size_t n=0;
	list_data_ * d = list->d;
	while (d){
		d = d->next;
		++n;
	}
	return n;
}

void dplist_foreach(const dpstringlist_t * begin, void (*func)(const dpstring_t *) ){
	list_data_ * p = begin->d;
	while (p){
		func(&p->value);
		p = p->next;
	}
}

int dplist_containsdp(const dpstringlist_t * list, const dpstring_t * str){
	return dplist_containss(list,dpstring_toc(str),str->len);
}

int dplist_containss(const dpstringlist_t * list, const char * str, const size_t len){
	list_data_ * p = list->d;
	while (p){
		if (dpstring_cmps(&p->value,str,len)==0)
			return 1;
		p = p->next;
	}
	return 0;
}

void dplist_appenddp(dpstringlist_t * list, const dpstring_t * s){
	dplist_appends(list,dpstring_toc(s),s->len);
}

void dplist_appends(dpstringlist_t * list, const char * s, const size_t len){
	if (!list->d){
		dplist_inits(list,s,len);
	} else{
		list_data_ * node = (list_data_*)malloc(sizeof(*node));
		dpstring_inits(&node->value,s,len);
		list->last->next = node;
		node->prev = list->last;
		node->next = 0;
		list->last = node;
	}
}

void dplist_print(const dpstringlist_t * list){
	list_data_ * p = list->d;
	printf("(");
	while (p){
		dpstring_print(&p->value);
		if (p != list->last){
			printf(",");
		}
		p = p->next;
	}
	printf(")");
}

void dplist_remove_onedp(dpstringlist_t * list, const dpstring_t * s){
	dplist_remove_ones(list,dpstring_toc(s),s->len);
}

void dplist_remove_ones(dpstringlist_t * list, const char * s, const size_t len){
	list_data_ * node = get_node_internals_(list->d,s,len);
	detach_and_delete_(list,node);
}

void dplist_remove_alldp(dpstringlist_t * list, const dpstring_t * s){
	dplist_remove_alls(list, dpstring_toc(s), s->len);
}

void dplist_remove_alls(dpstringlist_t * list, const char * s, const size_t len){
	list_data_ * node = get_node_internals_(list->d,s,len);
	while(node) {
		detach_and_delete_(list,node);
		node = get_node_internals_(list->d,s,len);
	}
}

void dplist_replace_onedp(dpstringlist_t * list, const dpstring_t * x, const dpstring_t * y){
	dplist_replace_ones(list,dpstring_toc(x),x->len,dpstring_toc(y),y->len);
}
void dplist_replace_onedps(dpstringlist_t * list, const dpstring_t * x, const char * y, const size_t len){
	dplist_replace_ones(list,dpstring_toc(x),x->len,y,len);
}
void dplist_replace_alldp(dpstringlist_t * list, const dpstring_t * x, const dpstring_t * y){
	dplist_replace_alls(list,dpstring_toc(x),x->len,dpstring_toc(y),y->len);
}

void dplist_replace_alldps(dpstringlist_t * list, const dpstring_t * x, const char * y, const size_t len){
	dplist_replace_alls(list,dpstring_toc(x),x->len,y,len);
}

void dplist_replace_ones(dpstringlist_t * list, const char * x, const size_t lenx, const char * y, const size_t leny){
	list_data_ * node = get_node_internals_(list->d,x,lenx);
	if (node){
		dpstring_copys(&node->value,y,leny);
	}
}


void dplist_replace_alls(dpstringlist_t * list, const char * x, const size_t lenx, const char * y, const size_t leny){
	list_data_ * node = get_node_internals_(list->d,x,lenx);
	while (node){
		dpstring_copys(&node->value,y,leny);
		node = get_node_internals_(list->d,x,lenx);
	}
}

void dplist_join(const dpstringlist_t * list, dpstring_t * out, const char * sep, const size_t seplen){
	dpstring_t result;
	list_data_ * node = list->d;
	dpstring_init(&result);
	while(node){
		dpstring_appenddp(&result,&node->value);
		if (seplen && (node != list->last)){
			dpstring_appends(&result,sep,seplen);
		}
		node = node->next;
	}
	dpstring_take(out,&result);
}
