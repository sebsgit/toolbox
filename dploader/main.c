#include <stdio.h>
#include <stdlib.h>
#include "dploader.h"


/*
 * test library loader by creating a .c source file with function definition
 * compile this .c source to shared library
 * attempt to load and execute the function
 * */
static void test_lib(){
    const char * path = "test.c";
    FILE * fp = fopen(path,"w");
    if (fp){
        const char buff[] = "int func(){ return 567; }";
        fwrite(buff,1,sizeof(buff)-1,fp);
        fclose(fp);
        system("gcc -shared -fPIC test.c -o test.lib");

        void * handle = dp_load("./test.lib");
        if (handle){
            typedef int (*fun_t)(void);
            fun_t f = dp_symbol(handle,"func");
            if (f){
                const int result = f();
                printf("result is: %i\n",result);
                if (result == 567){
                    remove("test.c");
                    remove("test.lib");
                } else{
                    printf("result error: %i\n",result);
                }
            } else{
                printf("symbol not found!");
            }
            dp_close(handle);
        } else{
            printf("cannot open library!\n");
        }
    } else{
        printf("cannot open test.c for writing!\n");
    }
}

int main(int argc, char ** argv){
    test_lib();
	return 0;
}
