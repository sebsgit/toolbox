#include <stdio.h>
#include <time.h>
#include "mempool.h"

static void double_free_handler(void* block) {
	printf("double free at address: %lu\n", (long)block);
}

static void test_pool() {
	vpool_init(512 * 1024);
	vpool_set_dobule_free_handler(double_free_handler);
	double* d = (double*)vpool_malloc(sizeof(*d));
	*d = 123;
	vpool_free(d);
	vpool_free(d);
	vpool_cleanup();
}

static void benchmark_frequent_allocs() {
	srand(time(0));
    const size_t alloc_size = 1024*1024*64;
	const int num_allocations = 1000;
	int count=0;
	clock_t start = clock();
	while (++count < num_allocations) {
		char* ptr = (char*)malloc(alloc_size / count*10);
		char* ptr1 = (char*)malloc(alloc_size / count*10);
		char* ptr2 = (char*)malloc(alloc_size / count*10);
		ptr[0] = 'c';
		ptr2[0] = 'a';
		ptr1[0] = 'x';
		free(ptr);
		free(ptr1);
		free(ptr2);
	}
	start = clock() - start;
	printf("malloc / free: %f\n", (1000.0f*start)/CLOCKS_PER_SEC);

	vpool_init(alloc_size);
	count=0;
	start = clock();
	while (++count < num_allocations) {
		char* ptr = (char*)vpool_malloc(alloc_size / count*10);
		char* ptr1 = (char*)vpool_malloc(alloc_size / count*10);
		char* ptr2 = (char*)vpool_malloc(alloc_size / count*10);
		ptr[0] = 'c';
		ptr2[0] = 'a';
		ptr1[0] = 'x';
		vpool_free(ptr);
		vpool_free(ptr1);
		vpool_free(ptr2);
	}
	start = clock() - start;
	printf("pool: %f\n", (1000.0f*start)/CLOCKS_PER_SEC);

	vpool_cleanup(alloc_size);
}

int main(int argc, char ** argv){
	test_pool();
	benchmark_frequent_allocs();
	return 0;
}
