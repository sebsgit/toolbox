extern "C"{

__global__ void kernel(int * value){
	*value = 5;
}

__global__ void kernel_2(unsigned int * values, unsigned int * value_count){
	if (threadIdx.x < *value_count){
		values[threadIdx.x] *= 2;
	}
}

}
