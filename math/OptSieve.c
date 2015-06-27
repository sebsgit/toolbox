#include <stdio.h>
#include <inttypes.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

/*
 * optimized prime sieve - do not keep multiples of 2 and 3
 * saves 66,666% space (1/2 - numbers divisible by 2, 1/6 - numbers divisible by 3 and not by 2)
 * 
 * *** reminder note to myself: how I came up with indexing functions:
 *   sieve: 5 7 11 13 17 19 23 25 ...
 * 
 * 	 1) we can split the above sequence into two sub-seqs:
 * 			p) 5 11 17 23 ...
 * 			q) 7 13 19 25 ...
 * 		both sequences have progress equal to six, so they are both of the form
 * 			y = 6*x + b
 * 		look at the indices in sieve to work out x and b
 * 			p) 0 2 4 6 ...
 * 			q) 1 3 5 7
 * 		so we can observe that x=i/2 for (p), and x=[i+1]/2 for (q)
 * 		which gives x=[i+i%2]/2 for both sequences
 * 
 * 		now figuring out b was straightforward (try some values for y):
 * 			in (p) we must always add 5, in (q) always add 1
 * 			so it is always b = 1 + 4*(1-i%2)
 * 		final equation:
 * 			y = 6*( [i+i%2]/2 ) + 1 + 4*(1 - i%2)
 * 
 *	2) figuring out the index in the sieve when given a number is a reverse of above process
 * 		first we observe that the sieve does not contain any multiples of 2 and 3, so we can skip those numbers
 * 		in fact I just worked out the general equation by writing the sieve sequence, writing the n/6 and n%3, n%2
 * 		parts above it and trying to put that together into a general equation
 * 
 * 		   n:	5  7  11  13  17  19 ...
 * 	 (n+1)/6:	1  1   2   2   3   3 ...
 *  	 n%3:	2  1   2   1   2   1 ...
 * 	 (n%3/2):  	1  0   1   0   1   0 ...
 * 1-(n%3/2):	0  1   0   1   0   1 ...
 * 
 *  and then just try to fit the pieces together
 * 	take a piece of paper and a pencil and you can easily figure out the equations
 * */

typedef uint64_t lint_t;

#ifndef SIEVE_BUFF_PART_BYTES
#define SIEVE_BUFF_PART_BYTES 1024*1024*16
#endif

/* returns ith number in sieve buffer 
 * reverse of get_index
 * get_index( get_number(i) ) == i
 * */
static lint_t get_number(const lint_t i){
	const short x = i%2;
	return 6*( (i+x)/2 ) +1 +4*(1-x);
}

/* returns index of n in sieve buffer 
 * reverse of get_number
 * get_number( get_index(n) ) == n
 * */
static lint_t get_index(const lint_t n){
	if (n%2==0){
		return 0;
	}else{
		const short n3 = n%3;
		if (n3 == 0)
			return 0;
		return 2*((n+1)/6) - 2 + (1-(n3)/2);
	}
	return 0;
}


struct buff_part_t{
	lint_t offset;
	char * data;
};

typedef struct buff_part_t buff_part_t;

static void buff_set_bit( buff_part_t * buff, const lint_t n ){
	const lint_t byte_num = n/8;
	const uint8_t mask = (1 << (n%8));
	*(uint8_t*)(buff->data+byte_num) |= mask;
}

static void buff_clear_bit(buff_part_t * buff, const lint_t n){
	const lint_t byte_num = n/8;
	const uint8_t mask = ~(1 << (n%8));
	*(uint8_t*)(buff->data+byte_num) &= mask;
}

static int buff_get_bit(buff_part_t * buff, const lint_t n){
	const lint_t byte_num = n/8;
	const uint8_t mask = (1 << (n%8));
	return (*(uint8_t*)(buff->data+byte_num) & (mask)) ? 1 : 0;
}

static void print_part(const buff_part_t * b, const lint_t size){
	lint_t i=0;
	for ( ; i<size ; ++i){
		const uint8_t val = *(uint8_t*)(b->data+i);
		lint_t n=0;
		for ( ; n<8 ; ++n){
			printf("%i",(val & (1 << n)) ? 1 : 0);
		}
		printf(" ");
	}
}

struct buff_t{
	struct buff_t * next;
	buff_part_t part;
	lint_t part_size;
	lint_t bit_count;
};

typedef struct buff_t buff_t;

static buff_t * alloc_private_(const lint_t bit_count, lint_t part_size, const lint_t offset, const int init_value){
	if (bit_count==0){
		return 0;
	} else{
		buff_t * b = malloc(sizeof(*b));
		if (bit_count < part_size*8)
			part_size = (bit_count+8)/8;
		b->part_size = part_size;
		b->part.offset = offset;
		b->part.data = malloc(part_size);
		if (!b->part.data){
			printf("malloc failed for size %lu!\n",part_size);
			exit(0);
		}
		/* we need bit pattern of 1's, not the number '1' */
		memset(b->part.data,init_value ? ~0 : 0,part_size);
		b->bit_count = bit_count;
		if (part_size*8 > bit_count){
			b->next = 0;
		} else{
			b->next = alloc_private_(bit_count - part_size*8, part_size, offset+part_size*8,init_value);
		}
		return b;
	}
}

static buff_t * alloc_buff(const lint_t bit_count, const lint_t part_size, int init_value){
	return alloc_private_(bit_count,part_size,0,init_value);
}

static void free_buff(buff_t * b){
	while (b){
		buff_t * next = b->next;
		free(b->part.data);
		free(b);
		b = next;
	}
}

static void set_bit(buff_t * b, const lint_t n){
	const lint_t part_num = n/(b->part_size*8);
	lint_t i=0;
	while (part_num > i && b){
		b = b->next;
		++i;
	}
	if (b){
		buff_set_bit(&b->part,n - b->part.offset);
	}
}

static int get_bit(buff_t * b, const lint_t n){
	const lint_t part_num = n/(b->part_size*8);
	lint_t i=0;
	while (part_num > i && b){
		b = b->next;
		++i;
	}
	if (b){
		return buff_get_bit(&b->part,n - b->part.offset);
	}
	return -1;
} 

static void clear_bit(buff_t * b, const lint_t n){
	const lint_t part_num = n/(b->part_size*8);
	lint_t i=0;
	while (part_num > i && b){
		b = b->next;
		++i;
	}
	if (b){
		buff_clear_bit(&b->part,n - b->part.offset);
	}
}

static void print_buff(const buff_t * b){
	while (b){
		print_part(&b->part,b->part_size);
		b = b->next;
	}
	printf("\n");
}

struct prime_sieve_t{
	lint_t max_number;
	buff_t * bitbuff;
};

typedef struct prime_sieve_t prime_sieve_t;

static void sieve_init_priv_(prime_sieve_t * sieve, const lint_t max_num){
	lint_t n = max_num;
	lint_t bits = get_index(n);
	while (bits == 0){
		++n;
		bits = get_index(n);
	}
	sieve->max_number = max_num;
	
	printf("%lu kb needed for prime sieve up to %lu\n",bits/(8*1024),max_num);
	
	sieve->bitbuff = alloc_buff(bits,SIEVE_BUFF_PART_BYTES,1);
	n=5;
	for ( ; n<=max_num ; ++n){
		lint_t idx = get_index(n);
		if ((idx > 0 || n==5) && (get_bit(sieve->bitbuff,idx) != 0)){
			lint_t mp = n*2;
			while (mp <= max_num){
				idx = get_index(mp);
				if ((idx > 0) && (get_bit(sieve->bitbuff,idx) != 0)){
					clear_bit(sieve->bitbuff,idx);
				}
				mp += n;
			}
		}
	}
}

static prime_sieve_t * alloc_sieve(const lint_t max_num){
	prime_sieve_t * sieve = (prime_sieve_t*)malloc(sizeof(*sieve));
	sieve_init_priv_(sieve,max_num);
	return sieve;
}

static void free_sieve(prime_sieve_t * sieve){
	if (sieve){
		free_buff(sieve->bitbuff);
		free(sieve);
	}
}

static int is_prime(const prime_sieve_t * sieve, const lint_t num){
	if (num <= 5){
		return num==2 || num==3 || num==5;
	} else if (num > sieve->max_number){
		return -1;
	} else {
		const lint_t idx = get_index(num);
		return idx > 0 && get_bit(sieve->bitbuff,get_index(num)) == 1;
	}
}

static void test_index(){
	lint_t i=0;
	for ( ; i<100000 ; ++i){
		assert(i == get_index(get_number(i)));
	}
}

static void test_buff(){
	buff_t * b = alloc_buff(1573,16,0);
	lint_t i=0;
	for ( ; i<b->bit_count ; ++i){
		lint_t k=0;
		set_bit(b,i);
		assert( 1 == get_bit(b,i) );
		for ( ; k<b->bit_count ; ++k){
			if (k != i){
				assert( 0 == get_bit(b,k) );
			}
		}
		clear_bit(b,i);
	}
	free_buff(b);
	(void)print_buff;
}

static int slow_verify(const lint_t prime){
	if (prime==2){
		return 1;
	} else if (prime%2==0){
		return 0;
	} else{
		lint_t n=3;
		for ( ; n<= prime/2 ; n+=2){
			if (prime%n==0)
				return 0;
		}
		return 1;
	}
}

static void test_sieve(){
	const lint_t n = 250000;
	prime_sieve_t * sieve = alloc_sieve(n);
	lint_t i=2;
	for ( ; i<=n ; ++i){
		assert( slow_verify(i) == is_prime(sieve,i) );
	}
	free_sieve(sieve);
}

int main(int argc, char ** argv){	
	test_index();
	test_buff();
	test_sieve();
	return 0;
}
