#include "thpool.h"
#include <stdio.h>
#include <windows.h>
#include "dyncall.h"

void func(){
    Sleep(100);
    printf("task executed 1\n");
}
void func2(){
    Sleep(190);
    printf("task executed 2\n");
}
void func3(){
    Sleep(140);
    printf("task executed 3\n");
}
void func4(){
    Sleep(100);
    printf("task executed 4\n");
}
void func5(){
    Sleep(105);
    printf("task executed 5\n");
}

void test_f(int a, char c){
    printf("test_f %i %c\n",a,c);
}

static void test(){
	int x=11;
	char c = 'D';
    thpool_init_default();
    thpool_exec_default(func);
    thpool_exec_default(func2);
    thpool_exec_default(func3);
    thpool_exec_default(func4);
    thpool_exec(THPOOL_EXEC_ABORT,func);
    thpool_exec(THPOOL_EXEC_ABORT,func2);
	
    thpool_stack_reset();
    thpool_stack_pushi(x);
    thpool_stack_pushc(c);
    thpool_exec_default((thtask_t)test_f);
	
    thpool_stack_reset();
    thpool_exec_default(func3);
    thpool_exec_default(func4);
    thpool_cleanup();
}

int main(){
    int x=1000;
    while (--x){
        test();
    }
	return 0;
}
