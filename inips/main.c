#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "inips.h"

static void test_ini(){
    inips_handle_t fh;
    dpstringlist_t keys;
    dpstring_t tmp;
    dpstring_init(&tmp);
    dplist_init(&keys);
    assert( ! inips_load_file("nosuchfile",&fh) );
    assert( inips_load_file("test.ini",&fh) );
    inips_get_keys(&fh,&keys);

    assert( inips_has_key(&fh,"test_key") );
    assert( inips_has_key(&fh,"keysd") );
    assert( inips_has_key(&fh,"key3") );
    assert( inips_has_key(&fh,"keyint") );
    assert( inips_has_key(&fh,"key4") );
    assert( inips_has_key(&fh,"test_double") );
    assert( ! inips_has_key(&fh,"nosuchkey") );
    assert( ! inips_has_key(&fh,"does not exist") );
    assert( ! inips_has_key(&fh,"key5") );
    assert( dplist_length(&keys) == 6 );
    assert( inips_get_int(&fh,"keyint") == 85 );
    assert( inips_get_double(&fh,"test_double") == -1.23 );
    inips_get_string(&fh,"key3",&tmp);
    assert( strncmp("string value quoted",dpstring_toc(&tmp),19) == 0 );

    inips_cleanup(&fh);
    dplist_cleanup(&keys);
    dpstring_cleanup(&tmp);
}

int main(int argc, char ** argv){
    size_t n=1000;
    while (n-- > 0){
        test_ini();
    }
	return 0;
}
