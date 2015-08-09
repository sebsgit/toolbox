#include <iostream>
#include <string>
#include <cassert>
#include <thread>
#include <cstring>

#include "cuwrap.h"

#define cuassert(what) { const cuwr::result_t err = what; if(err!=0){ std::cout << __LINE__ << ":" << cuwr::tostr(err); assert(0);} }

static const std::string kernel_source =
".version 4.2\n"
".target sm_20\n"
".address_size 64\n"
".visible .entry kernel(\n"
"	.param .u64 kernel_param_0\n"
")\n"
"{\n"
"	.reg .s32 	%r<2>;\n"
"	.reg .s64 	%rd<3>;\n"
"\n"
"\n"
"	ld.param.u64 	%rd1, [kernel_param_0];\n"
"	cvta.to.global.u64 	%rd2, %rd1;\n"
"	mov.u32 	%r1, 5;\n"
"	st.global.u32 	[%rd2], %r1;\n"
"	ret;\n"
"}";

static int stream_cb_launched = 0;

static void stream_cb(cuwr::stream_t /*s*/, cuwr::result_t res, void *){
    stream_cb_launched = 1;
    cuassert(res);
}

static void test_stream(){
    stream_cb_launched = 0;
    cuwr::stream_t stream=0;
    cuassert( cuwr::cuStreamCreate(&stream,cuwr::CU_STREAM_DEFAULT) );
    unsigned int flags=123;
    cuassert( cuwr::cuStreamGetFlags(stream,&flags) );
    assert(flags == cuwr::CU_STREAM_DEFAULT);

    cuassert(cuwr::cuStreamAddCallback(stream,stream_cb,0,0));
    cuwr::Module module("kernel.ptx");
    assert(module.isLoaded());
    cuwr::KernelLaunchParams params;
    params.setBlockSize(1,1);
    params.setStream(stream);
    cuwr::DeviceValue<int> value;
    params.push(value);
    cuassert( cuwr::launch_kernel(module.function("kernel"),params) );
    cuassert( cuwr::cuStreamSynchronize(stream) );
    assert(stream_cb_launched);
    cuassert( cuwr::cuStreamDestroy(stream) );
}

static void test_pinned_mem(cuwr::Gpu * gpu){
    if (gpu->attribute(cuwr::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY_)){
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

        unsigned int test = 19;
        cuwr::DeviceMemPinnedAllocator::pointer_type pinnedPtr;
        cuwr::DeviceMemPinnedAllocator::zero(&pinnedPtr);
        cuassert(cuwr::DeviceMemPinnedAllocator::alloc(&pinnedPtr,sizeof(test)));
        cuassert(cuwr::DeviceMemPinnedAllocator::copyToDevice(pinnedPtr,&test,sizeof(test)));
        test = 12323;
        cuassert(cuwr::DeviceMemPinnedAllocator::copyToHost(&test,pinnedPtr,sizeof(test)));
        assert(test == 19);

        cuwr::DeviceValue<int, cuwr::DeviceMemPinnedAllocator> val;
        val = 123;
        const int testVal = val;
        assert(val == testVal);

        params.clear();
        params.setBlockSize(1,1);
        params.push(val);
        cuassert(cuwr::launch_kernel(module.function("kernel"),params));
        unsigned int testVal2=1233;
        testVal2 = val;
        assert(testVal2 == 5);

        cuassert(cuwr::DeviceMemPinnedAllocator::free(pinnedPtr));
    } else{
        std::cout << gpu->name() << " : can't map host memory.\n";
    }
}

static void test_kernel(cuwr::Gpu * gpu){
    (void)gpu;
    cuwr::Timer timer;
    timer.start();

    unsigned int value=0;
    cuwr::DeviceValue<unsigned int, cuwr::DeviceMemPinnedAllocator> dptr;
	cuwr::Module module("kernel.ptx");
	assert(module.isLoaded());
	cuwr::KernelLaunchParams params;
	params.setBlockSize(2,2);
	params.push(dptr);

    cuassert(cuwr::launch_kernel(module.function("kernel"),params));
    value = dptr;

	assert( value == 5 );

    cuwr::DeviceValue<unsigned int> dev_count = 32;
    cuwr::DeviceArray<unsigned int, cuwr::DeviceMemPinnedAllocator> dev_array;
    unsigned int array[32];
	for (int i=0 ; i<32 ; ++i){
		array[i] = i+1;
	}
    dev_array.resize(sizeof(array)/sizeof(array[0]));
    dev_array.load(array);
	params.clear();
    params.push(dev_array);
	params.push(dev_count);
	params.setBlockSize(56,1);
    cuassert( cuwr::launch_kernel(module.function("kernel_2"),params) );
    dev_array.store(array);
	for (int i=0 ; i<32 ; ++i){
		assert( array[i] == 2u*(i+1) );
	}
    params.clear();
    int to_set = 123, tmp = -1;
    cuwr::DeviceValue<int> to_get = 0;
    params.clear();
    params.push(&to_set);
    params.push(to_get);
    cuassert( cuwr::launch_kernel(module.function("kernel_3"),params));
    tmp = to_get;
    assert( tmp == to_set );

    timer.stop();
    std::cout << "kernels launched in: " << timer.elapsed() << "ms.\n";
}

static void test_load_ptx_source(){
	unsigned int value=0;
    cuwr::DeviceValue<unsigned int, cuwr::DeviceMemPinnedAllocator> dptr;
	cuwr::Module module(kernel_source,cuwr::Module::PtxSource);
	assert(module.isLoaded());
	cuwr::KernelLaunchParams params;
	params.setBlockSize(2,2);
	params.push(dptr);

    cuassert(cuwr::launch_kernel(module.function("kernel"),params));
    value = dptr;

	assert( value == 5 );
}

static void test_suite(cuwr::Gpu * gpu){
    gpu->makeCurrent();
    test_kernel(gpu);
    test_pinned_mem(gpu);
    test_stream();
    test_load_ptx_source();
}

int main(){
    cuwr::Gpu * gpu = &cuwr::defaultGpu();
    assert(gpu);
    int driverVersion=0;
    int deviceCount=0;
    cuassert(cuwr::cuDeviceGetCount(&deviceCount));
    cuassert(cuwr::cuDriverGetVersion(&driverVersion));
    assert(gpu->computeCapability() >= 1.0);
    assert(deviceCount>0);
    cuwr::Gpu badGpu(deviceCount+1);
    assert(badGpu.isInitialized()==false);
    std::cout << "CUDA driver version " << driverVersion << "\n";
    std::cout << gpu->name() << " (compute caps: " << gpu->computeCapability() << ")\n";
    std::cout << gpu->totalMemory()/(1024*1024.0) << " mb \n";
    std::cout << gpu->attribute(cuwr::CU_DEVICE_ATTRIBUTE_CLOCK_RATE_)/1024.0 << " Mhz\n";

    test_suite(gpu);
    std::thread th(test_suite,gpu);
    th.join();

    cuwr::cleanup();
    return 0;
}
