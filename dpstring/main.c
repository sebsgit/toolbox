#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "dpstring.h"
#include "dpstringlist.h"
#include "dpfileio.h"

static void test_string(){
	dpstring_t string;
	dpstring_t string2;
	dpstring_t longstring;
	dpstring_inits(&string,"test string",11);
	dpstring_initcc(&string2,'x',5);
	dpstring_inits(&longstring,"1234567890123456789012345678901234567890", 40);
	
	assert( !dpstring_is_empty(&string2) );
	assert( !dpstring_is_empty(&string) );
	assert( !dpstring_is_empty(&longstring) );
	assert( dpstring_toc(&string) != 0);
	assert( dpstring_toc(&string2) != 0);
	assert( dpstring_toc(&longstring) != 0 );
	assert( strncmp("test string",dpstring_toc(&string), 11) == 0 );
	assert( strncmp("xxxxx",dpstring_toc(&string2), 5) == 0 );
	assert( dpstring_length(&string2) == 5 );
	assert( dpstring_length(&string) == 11 );
	assert( dpstring_getc(&string2,0) == 'x' );
	assert( dpstring_getc(&string2,1) == 'x' );
	assert( dpstring_getc(&string2,2) == 'x' );
	assert( dpstring_getc(&string2,3) == 'x' );
	assert( dpstring_getc(&string2,4) == 'x' );
	assert( dpstring_getc(&string,0) == 't' );
	assert( dpstring_getc(&string,1) == 'e' );
	assert( dpstring_getc(&string,2) == 's' );
	assert( dpstring_getc(&string,3) == 't' );
	assert( dpstring_getc(&string,4) == ' ' );
	assert( dpstring_getc(&string,5) == 's' );
	assert( dpstring_getc(&string,6) == 't' );
	assert( dpstring_getc(&string,7) == 'r' );
	assert( dpstring_getc(&string,8) == 'i' );
	assert( dpstring_getc(&string,9) == 'n' );
	assert( dpstring_getc(&string,10) == 'g' );
	assert( dpstring_locatec(&string,0,'j') == -1 );
	assert( dpstring_locatec(&string,0,'t') == 0 );
	assert( dpstring_locatec(&string,2,'t') == 3 );
	assert( dpstring_locatec(&string,4,'t') == 6 );
	assert( dpstring_locatec(&string,0,'g') == 10 );
	assert( dpstring_locatec(&longstring,0,'g') == -1 );
	dpstring_replacecc(&string2,'x','q');
	assert( strncmp("qqqqq", dpstring_toc(&string2), 5) == 0 );
	dpstring_copy(&string,&longstring);
	assert( dpstring_length(&string) == dpstring_length(&longstring) );
	assert( strncmp(dpstring_toc(&string), dpstring_toc(&longstring), dpstring_length(&string)) == 0 );
	dpstring_copys(&string,"  \t lots\t of\nwhitespace\r\n ",26);
	dpstring_trim(&string);
	assert( strncmp(dpstring_toc(&string),"lots\t of\nwhitespace",19) == 0 );
	
	dpstring_cleanup_fast(&string2);
	dpstring_cleanup(&string);
	dpstring_cleanup_fast(&longstring);
}

static void test_stringlist(){
	dpstringlist_t list;
	dpstring_t tmp;
	dpstringlist_iterator_t it;
	dplist_init(&list);
	dpstring_init(&tmp);
	dplist_appends(&list,"ala",3);
	dplist_appends(&list,"ma",2);
	dplist_appends(&list,"kota",4);
	
	assert( dplist_length(&list) == 3 );
	dplist_join(&list,&tmp,"",0);
	assert( strncmp("alamakota",dpstring_toc(&tmp),9) == 0 );
	dplist_join(&list,&tmp,";;",2);
	assert( strncmp("ala;;ma;;kota",dpstring_toc(&tmp),13) == 0 );
	
	dplist_beginv(&list,&it,&tmp);
	assert( strncmp("ala",dpstring_toc(&tmp),3) == 0 );
	assert( dplist_nextv(&it, &tmp) );
	assert( strncmp("ma",dpstring_toc(&tmp),2) == 0 );
	assert( dplist_nextv(&it, &tmp) );
	assert( strncmp("kota",dpstring_toc(&tmp),3) == 0 );
	assert( dplist_next(&it) == 0 );
	assert( dplist_is_valid(&it) == 0 );
	
	dplist_cleanup(&list);
	dpstring_copys(&tmp,"ola;tez;ma;kota",15);
	dplist_split(&tmp,&list,";",1);
	assert( dplist_length(&list) == 4 );
	
	dplist_cleanup(&list);
	dplist_split(&tmp,&list,";tez;",5);
	assert( dplist_length(&list) == 2 );
	
	dplist_cleanup(&list);
	dpstring_cleanup_fast(&tmp);
}

static void test_fileio(){
    const char * path = "test.txt";
    FILE * fp = fopen(path,"w");
    if (fp){
        dpstring_t tmp;
        dpstringlist_t list;
        dpstringlist_iterator_t it;
        fwrite("test\n",1,5,fp);
        fwrite("line2",1,5,fp);
        fclose(fp);

        dpstring_init(&tmp);
        dplist_init(&list);
        assert( dpio_load_file(path,&tmp) );
        assert( strncmp("test\nline2", dpstring_toc(&tmp), 10) == 0 );
        assert( dpio_split_file(path,"\n",1,&list) );
        assert( dplist_length(&list) == 2 );
        dplist_beginv(&list,&it,&tmp);
        assert( strncmp("test",dpstring_toc(&tmp),4) == 0 );
        assert( dplist_nextv(&it,&tmp) );
        assert( strncmp("line2",dpstring_toc(&tmp),5) == 0 );

        dpstring_copys(&tmp,"to write",8);
        dpio_writes(path,&tmp);
        dpstring_cleanup(&tmp);
        assert( dpstring_is_empty(&tmp) );
        dpio_load_file(path,&tmp);
        assert( strncmp("to write", dpstring_toc(&tmp), 8) == 0 );
        dpstring_copys(&tmp,"\nsecond",7);
        dpio_appends(path,&tmp);
        dpstring_cleanup(&tmp);
        assert( dpstring_is_empty(&tmp) );
        dpio_load_file(path,&tmp);
        assert( strncmp("to write\nsecond", dpstring_toc(&tmp), 15) == 0 );

        dpstring_cleanup_fast(&tmp);
        dplist_cleanup(&list);
        remove(path);
    } else{
        printf("can't open tmp file!\n");
        exit(0);
    }
}

int main(int argc, char ** argv){
	size_t n=1000;
	while (--n != 0){
		test_string();
		test_stringlist();
        test_fileio();
	}
	return 0;
}
