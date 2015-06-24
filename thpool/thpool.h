/*
 * Thread pool implementation.

 Thread pool api is meant to provide an easy way to call arbitrary function
 in separate thread. There is no need to manipulate low-level synchronization primitives
 or to launch a thread explicitly.

 * Example use case:

 thpool_init_default(); // initialize the pool
 thpool_exec_default(my_function);  // launch my_function() in separate thread
 thpool_cleanup();      // wait for all threads to finish and cleans up allocated resources

 * Executing functions with parameters:
 Parameters are managed through internal parameter stack. Default stack size is 16 parameters,
 up to 8 bytes each. This can be changed by recompiling thread pool with THPOOL_MAX_PARAM_COUNT
 macro redefined.
 This is how you can call function "void my_func(int,double,void *);":

 thpool_init_default();
 thpool_stack_reset();  // cleans up stack parameters from previous calls
 thpool_stack_pushi(56);
 thpool_stack_pushd(3.14);
 thpool_stack_pushvp((void*)p);
 thpool_exec_default(my_func);  // call my_func(56,3.14,p); in separate thread
 thpool_cleanup();

 If stack_autoreset is enabled, then parameters will be popped from the stack after each function call.
 Parameters should be pushed in the order of declaration.

 thpool_init_default();
 thpool_stack_reset();
 thpool_stack_autoreset(0);
 thpool_stack_pushi(56);
 thpool_stack_pushd(3.14);
 thpool_stack_pushvp((void*)p);
 thpool_exec_default(my_func);  // call my_func(56,3.14,p);
 thpool_exec_default(my_func);  // call my_func(56,3.14,p); again
 thpool_exec_default(my_func);  // call my_func(56,3.14,p); again

 thpool_stack_autoreset(1);
 thpool_exec_default(my_func);  // call my_func(56,3.14,p); and clears the stack
 thpool_exec_default(my_func);  // ERROR, no parameters on the stack, behavior undefined

 thpool_cleanup();

*/

#ifndef THPOL_HEADER_H
#define THPOL_HEADER_H

#ifdef __cplusplus
extern "C"{
#endif

#include <stdlib.h>

#define THPOOL_EXEC_BLOCK (1U << 0)     /* block the call until thread is available */
#define THPOOL_EXEC_ABORT (1U << 1)     /* abort the call if no thread is available */

typedef int ththread_id_t;
typedef unsigned thpool_exec_flags;

typedef void (*thtask_t)();

/*
 * initialize thread pool with default parameters
 * best number of threads is automatically detected
 * function parameter stack size is set to THPOOL_MAX_PARAM_COUNT*8 bytes (default 128)
 *
 * thpool_cleanup must be called to cleanup the pool resources
 */
extern void thpool_init_default();
/*
 * initialize thread pool with maximum thread count
 *
 * thpool_cleanup must be called to cleanup the pool resources
 */
extern void thpool_init(size_t n_threads);
/*
 * cleans up thread pool resources, waits for all threads to complete
 */
extern void thpool_cleanup();
/*
 * wait for all running thread to complete
 */
extern void thpool_wait();

/*
 * reset the parameter stack, discards previously pushed parameters
 * */
extern void thpool_stack_reset();
/*
 * use to switch parameter stack autoreset
 * if enabled, stack will be cleaned after each function call
 * */
extern void thpool_stack_autoreset(int enabled);
extern void thpool_stack_pushc(char c);
extern void thpool_stack_pushi(int x);
extern void thpool_stack_pushs(short x);
extern void thpool_stack_pushd(double x);
extern void thpool_stack_pushf(float x);
extern void thpool_stack_pushu(unsigned x);
extern void thpool_stack_pushl(long x);
extern void thpool_stack_pushll(long long x);
extern void thpool_stack_pushul(unsigned long x);
extern void thpool_stack_pushull(unsigned long long x);
/*
 * push p by value, do not delete p before the scheduled function is done
 * */
extern void thpool_stack_pushvp(void * p);

/*
 * execute a function with default flags (block until thread is available)
 * */
extern ththread_id_t thpool_exec_default(thtask_t func);
/*
 * execute a function with given flags
 * */
extern ththread_id_t thpool_exec(thpool_exec_flags flags, thtask_t func);

#ifdef __cplusplus
}
#endif

#endif
