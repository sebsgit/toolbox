#include "thpool.h"
#include <pthread.h>
#include "dyncall.h"

#ifndef THPOOL_MAX_PARAM_COUNT
#define THPOOL_MAX_PARAM_COUNT 16
#endif

enum ThpoolParamType{
	THPOOL_CHAR,
	THPOOL_SHORT,
	THPOOL_INT,
	THPOOL_UINT,
	THPOOL_LONG,
	THPOOL_LONGLONG,
	THPOOL_ULONG,
	THPOOL_ULONGLONG,
	THPOOL_FLOAT,
	THPOOL_DOUBLE,
	THPOOL_VOIDP
};
union ThpoolParamStorage{
	char c;
	short s;
	int i;
	unsigned u;
	long l;
	long long ll;
	unsigned long ul;
	unsigned long long ull;
	float f;
	double d;
	void * vp;
};

struct thparam_t{
	union ThpoolParamStorage store;
	enum ThpoolParamType type;
};
typedef struct thparam_t thparam_t;

static thparam_t thglobal_param_stack_[THPOOL_MAX_PARAM_COUNT];
static size_t thglobal_stack_ptr_ = 0;

static pthread_mutex_t thglobal_mutex;
static size_t thglobal_active_thread_num_ = 0;
static size_t thglobal_max_threads_ = 4;
static volatile int thglobal_stack_autoreset_;

struct thcallvm_t{
	struct thcallvm_t * next;
	DCCallVM * vm;
	pthread_mutex_t mutex;
	pthread_mutex_t busy_mutex;
	volatile int is_busy;
	volatile int is_scheduled;
};
typedef struct thcallvm_t thcallvm_t;

struct ththread_t{
	struct ththread_t * next;
	thcallvm_t * call_vm;
	ththread_id_t id;
	pthread_t thread;
	thtask_t func;
};
typedef struct ththread_t ththread_t;

static ththread_t * thglobal_queue;
static thcallvm_t * thglobal_vm;

static void push_params_(thcallvm_t * vm);

static void thread_init_(ththread_t * t){
	t->next = 0;
	t->call_vm = 0;
	t->func = 0;
	t->id = -1;
}

static void thread_destroy_(ththread_t * t){
	ththread_t * prev = 0;
	ththread_t * next = thglobal_queue;
	while (next){
		if (t == next){
			if (prev){
				prev->next = next->next;
			}
            if (t == thglobal_queue){
                thglobal_queue = thglobal_queue->next;
            }
			free(t);
			break;
		}
		prev = next;
		next = next->next;
	}
}

static thcallvm_t * new_callvm(size_t stack_size){
	thcallvm_t * vm = (thcallvm_t*)malloc(sizeof(*vm));
	vm->next = 0;
	vm->is_busy = 0;
	vm->is_scheduled = 0;
	pthread_mutex_init(&vm->mutex,0);
	pthread_mutex_init(&vm->busy_mutex,0);
	vm->vm = dcNewCallVM(stack_size);
	dcMode(vm->vm, DC_CALL_C_DEFAULT);
	dcReset(vm->vm);
	return vm;
}

void thpool_init_default(){
    return thpool_init(4);
}

void thpool_init(size_t n_threads){
	const size_t stack_size = (THPOOL_MAX_PARAM_COUNT*8) > 0 ? (THPOOL_MAX_PARAM_COUNT*8) : 1;
    thglobal_stack_autoreset_ = 0;
	thglobal_stack_ptr_ = 0;
	pthread_mutex_init(&thglobal_mutex,0);
	thglobal_active_thread_num_ = 0;
	thglobal_queue = 0;
	thglobal_vm = new_callvm(stack_size);
	size_t curr=1;
	thglobal_max_threads_ = n_threads;
	thcallvm_t * vm = thglobal_vm;
	while (curr < n_threads){
        thcallvm_t * node = new_callvm(stack_size);
        vm->next = node;
        vm = vm->next;
		++curr;
	}
}

static void kill_all_(){
    pthread_mutex_lock(&thglobal_mutex);
	while (thglobal_vm){
		pthread_mutex_destroy(&thglobal_vm->mutex);
		pthread_mutex_destroy(&thglobal_vm->busy_mutex);
        dcReset(thglobal_vm->vm);
		dcFree(thglobal_vm->vm);
		thcallvm_t * t = thglobal_vm->next;
		free(thglobal_vm);
		thglobal_vm = t;
	}
	while (thglobal_queue){
		ththread_t * t = thglobal_queue->next;
        pthread_detach(thglobal_queue->thread);
		pthread_cancel(thglobal_queue->thread);
		free(thglobal_queue);
		thglobal_queue = t;
    }
    pthread_mutex_unlock(&thglobal_mutex);
    pthread_mutex_destroy(&thglobal_mutex);
}

void thpool_wait(){
	thcallvm_t * vm = thglobal_vm;
	while (vm){
		pthread_mutex_lock(&thglobal_mutex);
		if (thglobal_active_thread_num_ > 0){
			pthread_mutex_unlock(&thglobal_mutex);
			sched_yield();
			continue;
		}
		int is_busy = 0;
		pthread_mutex_lock(&vm->busy_mutex);
		if (vm->is_busy || vm->is_scheduled){
			is_busy = 1;
		}
		pthread_mutex_unlock(&vm->busy_mutex);
		if (is_busy){
			vm = thglobal_vm;
		} else{
			vm = vm->next;
		}
		pthread_mutex_unlock(&thglobal_mutex);
		sched_yield();
	}
}

void thpool_cleanup(){
	thpool_wait();
	kill_all_();
}

static void * thread_proc_(void * p){
	pthread_mutex_lock(&thglobal_mutex);
	++thglobal_active_thread_num_;
	pthread_mutex_unlock(&thglobal_mutex);
	
	ththread_t * th = (ththread_t*)p;
	if (th){
		pthread_mutex_lock(&th->call_vm->mutex);
        dcCallVoid(th->call_vm->vm, (DCpointer)th->func);
        if (thglobal_stack_autoreset_){
            dcReset(th->call_vm->vm);
        }
		pthread_mutex_unlock(&th->call_vm->mutex);
		
		pthread_mutex_lock(&th->call_vm->busy_mutex);
		th->call_vm->is_busy = 0;
		th->call_vm->is_scheduled = 0;
		pthread_mutex_unlock(&th->call_vm->busy_mutex);
	}
	
	pthread_mutex_lock(&thglobal_mutex);
	--thglobal_active_thread_num_;
    pthread_detach(pthread_self());
	thread_destroy_(th);
	pthread_mutex_unlock(&thglobal_mutex);
	return 0;
}

static thcallvm_t * get_scheduled_vm_(){
	thcallvm_t * t = thglobal_vm;
	thcallvm_t * lazy_vm = 0;
	while (t){
		int is_scheduled=0;
		pthread_mutex_lock(&t->busy_mutex);
		is_scheduled = t->is_scheduled && !t->is_busy;
		if (lazy_vm==0 && t->is_busy==0 && t->is_scheduled==0){
			lazy_vm = t;
		}
		pthread_mutex_unlock(&t->busy_mutex);
		if (is_scheduled != 0){
			break;
		} else{
			t = t->next;
		}
	}
	if (t == 0 && lazy_vm){
		pthread_mutex_lock(&lazy_vm->busy_mutex);
		lazy_vm->is_scheduled=1;
		t = lazy_vm;
		pthread_mutex_unlock(&lazy_vm->busy_mutex);
	}
	return t;
}

static ththread_t * get_free_thread_(){
	pthread_mutex_lock(&thglobal_mutex);
    if (thglobal_active_thread_num_ < thglobal_max_threads_){
		ththread_t * th = (ththread_t*)malloc(sizeof(*th));
		thread_init_(th);
		if (thglobal_queue == 0){
			thglobal_queue = th;
			th->id = 1;
		} else{
			size_t id=1;
			ththread_t * node = thglobal_queue;
			while (node->next){
				++id;
				node = node->next;
			}
			node->next = th;
			th->id = id;
		}
        pthread_mutex_unlock(&thglobal_mutex);
		return th;
	} else{
        pthread_mutex_unlock(&thglobal_mutex);
		return 0;
	}
}

static ththread_id_t exec_internal_(thpool_exec_flags flags, thtask_t func){
	thcallvm_t * vm = get_scheduled_vm_();
	ththread_t * th = 0;
	if (!vm){
        if (flags & THPOOL_EXEC_ABORT){
			return -1;
        }else if (flags & THPOOL_EXEC_BLOCK){
			while (vm == 0){
				sched_yield();
				vm = get_scheduled_vm_();
			}
		}
	}
	if (vm){
		ththread_t * th = get_free_thread_();
		if (!th){
            if (flags & THPOOL_EXEC_ABORT){
				return -1;
            } if (flags & THPOOL_EXEC_BLOCK){
				while (th == 0){
					sched_yield();
					th = get_free_thread_();
				}
			}
		}
		if (th){
			th->func = func;
			th->call_vm = vm;
			pthread_mutex_lock(&vm->busy_mutex);
			vm->is_busy = 1;
			vm->is_scheduled = 0;
			push_params_(vm);
			pthread_mutex_unlock(&vm->busy_mutex);
            if (pthread_create(&th->thread,0,thread_proc_,th) != 0){
                if (thglobal_stack_autoreset_)
                    dcReset(vm->vm);
                thread_destroy_(th);
                return -1;
            }
		}
	}
	return th ? th->id : 0;
} 

ththread_id_t thpool_exec_default(thtask_t func){
    return exec_internal_(THPOOL_EXEC_BLOCK,func);
}

ththread_id_t thpool_exec(thpool_exec_flags flags, thtask_t func){
	return exec_internal_(flags,func);
}

// PARAMETER STACK MANAGEMENT

static void push_params_(thcallvm_t * vm){
	dcReset(vm->vm);
	size_t n_params = 0;
	for ( ; n_params < thglobal_stack_ptr_ ; ++n_params){
		const thparam_t t = thglobal_param_stack_[n_params];
		switch (t.type){
		case THPOOL_INT: dcArgInt(vm->vm, t.store.i); break;
		case THPOOL_CHAR: dcArgChar(vm->vm, t.store.c); break;
		case THPOOL_SHORT: dcArgShort(vm->vm,t.store.s); break;
		case THPOOL_UINT: dcArgInt(vm->vm,t.store.u); break;
		case THPOOL_LONG: dcArgLong(vm->vm,t.store.l); break;
		case THPOOL_LONGLONG: dcArgLongLong(vm->vm,t.store.ll); break;
		case THPOOL_ULONG: dcArgLong(vm->vm,t.store.ul); break;
		case THPOOL_ULONGLONG: dcArgLongLong(vm->vm,t.store.ull); break;
		case THPOOL_FLOAT: dcArgFloat(vm->vm,t.store.f); break;
		case THPOOL_DOUBLE: dcArgDouble(vm->vm,t.store.d); break;
		case THPOOL_VOIDP: dcArgPointer(vm->vm,t.store.vp); break;
		default:
		break;
		}
	}
}

void thpool_stack_reset(){
	thglobal_stack_ptr_ = 0;
}

void thpool_stack_autoreset(int enabled){
    thglobal_stack_autoreset_ = enabled;
}

#define PUSH_PARAM(x,t,f) thglobal_param_stack_[thglobal_stack_ptr_].store. f = x;  \
                        thglobal_param_stack_[thglobal_stack_ptr_].type = t;        \
						++thglobal_stack_ptr_;

void thpool_stack_pushc(char c){
    PUSH_PARAM(c,THPOOL_CHAR,c);
}

void thpool_stack_pushi(int x){
    PUSH_PARAM(x,THPOOL_INT,i);
}

void thpool_stack_pushs(short x){
    PUSH_PARAM(x,THPOOL_SHORT,s);
}

void thpool_stack_pushd(double x){
    PUSH_PARAM(x,THPOOL_DOUBLE,d);
}

void thpool_stack_pushf(float x){
    PUSH_PARAM(x,THPOOL_FLOAT,f);
}

void thpool_stack_pushu(unsigned x){
    PUSH_PARAM(x,THPOOL_UINT,u);
}

void thpool_stack_pushl(long x){
    PUSH_PARAM(x,THPOOL_LONG,l);
}

void thpool_stack_pushll(long long x){
    PUSH_PARAM(x,THPOOL_LONGLONG,ll);
}

void thpool_stack_pushul(unsigned long x){
    PUSH_PARAM(x,THPOOL_ULONG,ul);
}

void thpool_stack_pushull(unsigned long long x){
    PUSH_PARAM(x,THPOOL_ULONGLONG,ull);
}

void thpool_stack_pushvp(void * p){
    PUSH_PARAM(p,THPOOL_VOIDP,vp);
}

#undef PUSH_PARAM
