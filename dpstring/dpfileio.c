#include "dpfileio.h"
#include <stdio.h>

int dpio_load_file(const char *path, dpstring_t *out){
    FILE * fp = fopen(path,"r");
    if (fp){
        size_t size;
        fseek (fp, 0, SEEK_END);
        size = ftell(fp);
        if (size > 0){
            fseek(fp, 0, SEEK_SET);
            dpstring_resize(out,size);
            if( size != fread(dpstring_strbuf(out),1,size,fp) ){
				fclose(fp);
				return 0;
			}
        }
        fclose(fp);
        return 1;
    }
    return 0;
}

int dpio_split_file(const char *path, const char *sep, const size_t seplen, dpstringlist_t *out){
    int status=0;
    dpstring_t tmp;
    dpstring_init(&tmp);
    if (dpio_load_file(path,&tmp)){
        dplist_split(&tmp,out,sep,seplen);
        status = 1;
    }
    dpstring_cleanup_fast(&tmp);
    return status;
}

static void write_priv_(const char * path, const dpstring_t * str, const char * mode){
    FILE * fp = fopen(path,mode);
    if (fp){
        fwrite(dpstring_toc(str),1,str->len,fp);
        fclose(fp);
    }
}

void dpio_writes(const char * path, const dpstring_t * to_write){
    write_priv_(path,to_write,"w");
}

void dpio_appends(const char * path, const dpstring_t * to_append){
    write_priv_(path,to_append,"a");
}
