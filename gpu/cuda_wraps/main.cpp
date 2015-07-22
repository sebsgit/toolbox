#include <iostream>
#include <string>
#include <cassert>
#include <thread>
#include <cstring>

#include "cuwrap.h"

#define cuassert(what) { const cuwr::result_t err = what; if(err!=0){ std::cout << __LINE__ << ":" << cuwr::tostr(err); assert(0);} }

static void test_pinned_mem(cuwr::Gpu * gpu){
    if (gpu->canMapHost()){
        size_t freebytes, total;
        cuassert(cuwr::cuMemGetInfo(&freebytes,&total));
        assert(freebytes > 0);
        assert(total > 0);
        size_t nBytes = 1024*1024*128;
        void * mem = malloc(nBytes);
        void * hostMem = 0;
        cuwr::device_memptr_t devPtr=0;
        assert(mem);
        cuassert(cuwr::cuMemHostRegister(mem,nBytes,cuwr::CU_MEMHOSTREGISTER_PORTABLE_));
        cuassert(cuwr::cuMemHostAlloc(&hostMem,nBytes,cuwr::CU_MEMHOSTALLOC_DEVICEMAP_));
        memset(hostMem,0,nBytes);
        cuassert(cuwr::cuMemHostGetDevicePointer(&devPtr,hostMem,0));

        cuwr::Module module("kernel.ptx");
        assert(module.isLoaded());
        cuwr::KernelLaunchParams params;
        params.setBlockSize(1,1);
        params.push(&devPtr);

        assert(5 != *(unsigned int*)(hostMem));
        cuassert(cuwr::launch_kernel(module.function("kernel"),params));
        cuassert(cuwr::cuCtxSynchronize());

        assert(5 == *(unsigned int*)(hostMem));

        cuassert(cuwr::cuMemFreeHost(hostMem));
        cuassert(cuwr::cuMemHostUnregister(mem));
        free(mem);
    } else{
        std::cout << gpu->name() << " : can't map host memory.\n";
    }
}

static void test_kernel(cuwr::Gpu * gpu){
    cuwr::Timer timer;
    timer.start();

    unsigned int value=0;
	cuwr::DevicePtr<unsigned int> dptr;
	assert(dptr != 0);
	cuwr::Module module("kernel.ptx");
	assert(module.isLoaded());
	cuwr::KernelLaunchParams params;
	params.setBlockSize(2,2);
	params.push(dptr);

    cuassert(cuwr::launch_kernel(module.function("kernel"),params));
    dptr >> value;

	assert( value == 5 );
	
    cuwr::DevicePtr<unsigned int> dev_count = 32;
	unsigned int array[32];
	for (int i=0 ; i<32 ; ++i){
		array[i] = i+1;
	}
    dptr.realloc(sizeof(array));
    dptr << array;
	params.clear();
	params.push(dptr);
	params.push(dev_count);
	params.setBlockSize(56,1);
    cuassert( cuwr::launch_kernel(module.function("kernel_2"),params) );
    dptr >> array;
	for (int i=0 ; i<32 ; ++i){
		assert( array[i] == 2u*(i+1) );
	}
    params.clear();
    int to_set = 123, tmp = -1;
    cuwr::DevicePtr<int> to_get = 0;
    params.clear();
    params.push(&to_set);
    params.push(to_get);
    cuassert( cuwr::launch_kernel(module.function("kernel_3"),params));
    to_get.store(&tmp);
    assert( tmp == to_set );

    timer.stop();
    std::cout << "kernels launched in: " << timer.elapsed() << "ms.\n";
}

static void test_suite(cuwr::Gpu * gpu){
    gpu->makeCurrent();
    test_kernel(gpu);
    test_pinned_mem(gpu);
}

int main(){
	assert( cuwr::init() == 0 );
	cuwr::Gpu * gpu = new cuwr::Gpu();
    int driverVersion=0;
    cuassert(cuwr::cuDriverGetVersion(&driverVersion));
    std::cout << "CUDA driver version " << driverVersion << "\n";
	std::cout << gpu->name() << " (compute caps: " << gpu->computeCapabilityStr() << ")\n";
	std::cout << gpu->totalMemory()/(1024*1024.0) << " mb \n";
	std::cout << gpu->attribute(cuwr::CU_DEVICE_ATTRIBUTE_CLOCK_RATE_)/1024.0 << " Mhz\n";

    test_suite(gpu);
    std::thread th(test_suite,gpu);
	th.join();

	delete gpu;
	cuwr::cleanup();
	return 0;
}
