#include <iostream>
#include <string>
#include <cassert>
#include <thread>

#include "cuwrap.hpp"

static void test_kernel(cuwr::Gpu * gpu){
	gpu->makeCurrent();
    unsigned int value=0;
	cuwr::DevicePtr<unsigned int> dptr;
	assert(dptr != 0);
	cuwr::Module module("kernel.ptx");
	assert(module.isLoaded());
	cuwr::KernelLaunchParams params;
	params.setBlockSize(2,2);
	params.push(dptr);
	assert(0==cuwr::launch_kernel(module.function("kernel"),params));
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
	assert( 0 == cuwr::launch_kernel(module.function("kernel_2"),params) );
    dptr >> array;
	for (int i=0 ; i<32 ; ++i){
		assert( array[i] == 2u*(i+1) );
	}
}

int main(){
	assert( cuwr::init() == 0 );
	cuwr::Gpu * gpu = new cuwr::Gpu();
	std::cout << gpu->name() << " (compute caps: " << gpu->computeCapabilityStr() << ")\n";
	std::cout << gpu->totalMemory()/(1024*1024.0) << " mb \n";
	std::cout << gpu->attribute(cuwr::CU_DEVICE_ATTRIBUTE_CLOCK_RATE_)/1024.0 << " Mhz\n";
	std::cout << "can map host ? " << gpu->canMapHost() << "\n";

    test_kernel(gpu);
	std::thread th(test_kernel,gpu);
	th.join();
	
	delete gpu;
	cuwr::cleanup();
	return 0;
}
