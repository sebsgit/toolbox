#include "thpool.h"
#include <stdio.h>
#include <assert.h>

static void func(int a, int b, int * c){
	*c = a + b;
}

static void test(){
    thpool_init_default();
    int result=0;
    thpool_stack_push(21);
    thpool_stack_push(83);
    thpool_stack_push(&result);
    thpool_exec_default(func);
    thpool_wait();
    assert( 104 == result );
    thpool_cleanup();
}

int main(){
    int x=1000;
    while (--x){
        test();
    }
	return 0;
}
