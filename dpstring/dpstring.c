#include "dpstring.h"
#include <stdio.h>
#include <string.h>

#define IS_SHORT(str) ( (str)->buffsize == 0 )
#define STR_SRC(str) ( IS_SHORT((str)) ? (str)->data : (str->ptr) )
#define MAX_SHORT sizeof(str->data)

static size_t min(const size_t a, const size_t b){
	return a < b ? a : b;
}

#define CHECK_N(x) if (len < x) return x;

static size_t best_size_(const size_t len){
	if (len <16)
		return len;
	CHECK_N(32);
	CHECK_N(64);
	CHECK_N(128);
	CHECK_N(256);
	CHECK_N(512);
	CHECK_N(1024);
	CHECK_N(2048);
	return ((len/2048)*2048)+2048;
}

#undef CHECK_N

static void expand_(dpstring_t * str, const size_t len){
	const size_t s = best_size_(len);
	if (s < 16){
		if (IS_SHORT(str) == 0){
			memcpy(str->data, str->ptr, 16);
			str->buffsize=0;
			free(str->ptr);
			str->ptr = NULL;
		}
	} else{
		if (IS_SHORT(str) == 0){
			if (str->buffsize != s){
				char * new_buff = (char*)malloc(s);
				memcpy(new_buff, str->ptr, min(str->buffsize,s));
				free(str->ptr);
				str->ptr = new_buff;
				str->buffsize = s;
			}
		} else{
			str->buffsize = s;
			str->ptr = (char*)malloc(s);
			memcpy(str->ptr, str->data, str->len);
		}
	}
	STR_SRC(str)[len] = '\0';
	str->len = len;
}

void dpstring_init(dpstring_t * str){
	str->len=0;
	str->ptr=0;
	str->buffsize=0;
	memset(str->data,0,sizeof(str->data));
}

void dpstring_inits(dpstring_t * str, const char * buff, const size_t len){
	str->len = len;
	if (len < 16){
		str->ptr = NULL;
		str->buffsize=0;
	} else{
		str->buffsize = best_size_(len);
		str->ptr = (char*)malloc(str->buffsize);
	}
	memcpy(STR_SRC(str), buff, len);
	STR_SRC(str)[len] = 0;
}

void dpstring_initcc(dpstring_t * str, const char c, const size_t count){
	str->len = count;
	if (count < 16){
		str->ptr = NULL;
		str->buffsize=0;
	} else{
		str->buffsize = best_size_(count);
		str->ptr = (char*)malloc(str->buffsize);
		
	}
	memset(STR_SRC(str),c,count);
	STR_SRC(str)[count] = 0;
}

void dpstring_cleanup(dpstring_t * str){
	dpstring_cleanup_fast(str);
	str->buffsize=0;
	str->len=0;
	str->ptr = NULL;
	memset(str->data, 0, sizeof(str->data));
}

void dpstring_cleanup_fast(dpstring_t * str){
	if (IS_SHORT(str) == 0){
		free(str->ptr);
	}
}

int dpstring_is_empty(const dpstring_t * str){
    return str->len==0;
}

const uint32_t dpstring_length(const dpstring_t * str){
	return str->len;
}

const char dpstring_char_at(const dpstring_t * str, const size_t i){
	return STR_SRC(str)[i];
}

void dpstring_copy(dpstring_t * dest, const dpstring_t * src){
    dpstring_copys(dest,STR_SRC(src),src->len);
}

void dpstring_copys(dpstring_t * dest, const char * buff, const size_t len){
    expand_(dest,len);
    memcpy(STR_SRC(dest),buff,len);
}

void dpstring_store(char * buff, const dpstring_t * str, const size_t len){
	memcpy(buff,STR_SRC(str),min(len, str->len));
	buff[min(len,str->len)] = '\0';
}

void dpstring_print(const dpstring_t * str){
	printf("%s",STR_SRC(str));
}

void dpstring_appendc(dpstring_t * str, char c){
    expand_(str,str->len+1);
	STR_SRC(str)[str->len-1] = c;
}

void dpstring_appends(dpstring_t * str, const char * buff, const size_t len){
	const size_t prev_len = str->len;
	expand_(str,str->len+len);
    memcpy(STR_SRC(str)+prev_len,buff,len);
}

void dpstring_appenddp(dpstring_t * str, const dpstring_t * buff){
	dpstring_appends(str,STR_SRC(buff),buff->len);
}

int dpstring_cmp(const dpstring_t * str1, const dpstring_t * str2){
	return dpstring_cmps(str1,STR_SRC(str2),str2->len);
}

int dpstring_cmps(const dpstring_t * str1, const char * buff2, const size_t len){
	const size_t n = min(str1->len, len);
	const char * buff1 = STR_SRC(str1);
	size_t i=0;
	for ( ; i<n ; ++i){
		if (buff1[i] < buff2[i])
			return -1;
		else if (buff1[i] > buff2[i])
			return 1;
	}
	return str1->len < len ? -1 : (str1->len > len ? 1 : 0);
}

const char dpstring_getc(const dpstring_t * str, const size_t pos){
	return STR_SRC(str)[pos];
}

const char * dpstring_toc(const dpstring_t * str){
	return STR_SRC(str);
}

static int locate_internal_(const dpstring_t * str, const size_t pos, const char * buff, const size_t len){
	const char * src = STR_SRC(str);
	size_t i=pos;
	for ( ; i<str->len ; ++i ){
		if (src[i] == buff[0]){
			size_t tmp=1;
			for ( ; tmp<len ; ++tmp){
				if (i+tmp >= str->len)
					return -1;
				if (src[i+tmp] != buff[tmp])
					break;
			}
			if (tmp==len)
				return i;
		}
	}
	return -1;
}

int dpstring_locates(const dpstring_t * str, const size_t pos, const char * buff, const size_t len){
	return locate_internal_(str,pos,buff,len);
}

int dpstring_locatedp(const dpstring_t * str, const size_t pos, const dpstring_t * buff){
    return locate_internal_(str,pos,STR_SRC(buff),buff->len);
}

int dpstring_locatec(const dpstring_t * str, const size_t pos, const char c){
	const char * buff = STR_SRC(str);
	size_t i=pos;
	for ( ; i<str->len ; ++i){
		if (buff[i]==c)
			return i;
	}
	return -1;
}

void dpstring_replacecc(dpstring_t * str, const char x, const char y){
	char * buff = STR_SRC(str);
	size_t i=0;
	for ( ; i<str->len ; ++i){
		if (buff[i] == x)
			buff[i] = y;
	}
}

static void replace_internal_(dpstring_t * str, const char * x, const size_t xlen, const char * buff, const size_t bufflen, const size_t start_pos){
	const char * src = STR_SRC(str);
	dpstring_t result;
    size_t i=start_pos;
	dpstring_init(&result);
    if (start_pos > 0){
        dpstring_copys(&result,src,start_pos);
    }
    for (; i<str->len ; ++i){
		int copy_char=1;
		if (src[i] == x[0]){
			size_t tmp=1;
			for ( ; tmp < xlen ; ++tmp){
				if (i+tmp >= str->len)
					break;
				if (src[i+tmp] != x[tmp]){
					break;
				}
			}
			if (tmp == xlen){
				copy_char=0;
				dpstring_appends(&result,buff,bufflen);
				i += xlen-1;
			}
		}
		if (copy_char){
			dpstring_appendc(&result,src[i]);
		}
	}
	dpstring_take(str,&result);
}

void dpstring_replacecs(dpstring_t * str, const char x, const char * buff, const size_t len){
	char tmpbuff[2];
	tmpbuff[0] = x;
	tmpbuff[1] = '\0';
    replace_internal_(str,tmpbuff,1,buff,len,0);
}

void dpstring_replacess(dpstring_t * str, const char * x, const size_t lenx, const char * buff, const size_t lenbuff){
    replace_internal_(str,x,lenx,buff,lenbuff,0);
}

void dpstring_replacedp(dpstring_t * str, const dpstring_t * x, const dpstring_t * y){
    replace_internal_(str,STR_SRC(x),x->len,STR_SRC(y),y->len, 0);
}

void dpstring_replacedpi(dpstring_t * str, const dpstring_t * x, const dpstring_t * y, const size_t start_pos){
    replace_internal_(str,STR_SRC(x),x->len,STR_SRC(y),y->len, start_pos);
}

void dpstring_replacei(dpstring_t * str, const size_t a, const size_t b, const dpstring_t * y){
    if ((b > a) && (y->len>0)){
        dpstring_t result;
        dpstring_inits(&result,STR_SRC(str),a);
        dpstring_appenddp(&result,y);
        dpstring_appends(&result,STR_SRC(str)+(b+1),str->len-b);
        dpstring_take(str,&result);
    }
}

void dpstring_take(dpstring_t * str, dpstring_t * to_empty){
	if (IS_SHORT(to_empty)==0){
		/* take internal represenation */
		if (IS_SHORT(str)==0){
			free(str->ptr);
		}
		str->ptr = to_empty->ptr;
		str->len = to_empty->len;
		str->buffsize = to_empty->buffsize;
		to_empty->len=0;
		to_empty->buffsize=0;
		to_empty->ptr = NULL;
	} else{
		dpstring_copys(str,to_empty->data,to_empty->len);
	}
}

void dpstring_boundc(const dpstring_t * str, const char a, const char b, const size_t start, dpstring_t * out){
    const char * buff = STR_SRC(str);
    int posa =-1, posb=-1;
    size_t i=start;
    for ( ; i<str->len ; ++i){
        if (buff[i]==a && posa==-1)
            posa = i;
        if (buff[i]==b && posb==-1)
            posb = i;
        if (posa > -1 && posb > -1)
            break;
    }
    if (posa > -1 && posb > -1 && posb > posa){
        dpstring_cleanup(out);
        dpstring_copys(out,buff+posa+1,posb-posa-1);
    }
}
