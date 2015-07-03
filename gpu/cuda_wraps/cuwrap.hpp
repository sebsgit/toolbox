#ifndef CUWRAPS_HPP_
#define CUWRAPS_HPP_

/*
 * CUDA driver api wrapped in single header
 * created to allow distributing cuda apps without
 * linking to the nvidia libraries
 * typical use case will be to compile gpu code to .ptx
 * and distribute it as embedded resource with the app
 * plus a set of wrapper classes I find useful
 * */

//TODO stream
//TODO context
//TODO event
//TODO memory
//TODO module
//TODO host pinned mem ptr
//TODO autodetect best kernel launch size
//TODO nice demo (fractals ? they are always nice)

#include <cstdio>
#ifdef _WIN32
	#include <windows.h>
#elif defined __linux__
    #include <dlfcn.h>
#else
	#error Unsupported platform !
#endif
#include <functional>
#include <set>
#include <string>
#include <sstream>
#include <vector>

#define CUWR_NOCPY(K) private:						\
                        K (const K&) = delete;      \
                        K(K &&) = delete;           \
                        K& operator = (const K&) = delete;	\
                        K& operator = (K&&) = delete;

namespace cuwr{
	
	namespace priv{
		#ifdef _WIN32
			typedef HMODULE dplibhandle_priv_t_;
		#elif defined __linux__
			typedef void * dplibhandle_priv_t_;
		#endif
		
		static std::set<std::string> libcu_searchpath;
		static dplibhandle_priv_t_ libcu_handle;
		
		void * dp_load(const char * path){
			#ifdef _WIN32
				return LoadLibraryA(path);
			#elif defined __linux__
				dlerror();
				return dlopen(path,RTLD_LAZY | RTLD_LOCAL);
			#endif
		}

		void * dp_symbol(void * h, const char * sym){
			dplibhandle_priv_t_ p = (dplibhandle_priv_t_)h;
			#ifdef _WIN32
				return GetProcAddress(p,sym);
			#elif defined __linux__
				dlerror();
				return dlsym(p,sym);
			#endif
		}

		const char * dp_error(){
			#ifdef _WIN32
				static char buff[256];
				snprintf(buff,256,"ERROR CODE: %i\n",GetLastError());
				return buff;
			#elif defined __linux__
				return dlerror();
			#endif
		}

		void dp_close(void * h){
			dplibhandle_priv_t_ p = (dplibhandle_priv_t_)h;
			if (p){
				#ifdef _WIN32
					FreeLibrary(p);
				#elif defined __linux__
					dlclose(p);
				#endif
			}
		}
		
		int locate_cuda_rt(char * out_path, int maxLen){
			#ifdef __linux__
				const std::string lname = "libcuda.so";
			#elif defined __WIN32
				const std::string lname = "nvidia.dll";
			#endif
			for (const std::string& s : libcu_searchpath){
				const std::string full_path = s+lname;
				if (void * h = dp_load(full_path.c_str())){
					dp_close(h);
					snprintf(out_path,maxLen,"%s",full_path.c_str());
					return 0;
				}
			}
			return -1;
		}
		
		template <typename R, typename... Arg>
		void load_func(void * lib_handle, std::function< R (Arg...) > & fc, const char * fname){
			fc = (R (*)(Arg...))priv::dp_symbol(lib_handle,fname);
			if (!fc){
				std::cerr << "could not locate symbol '"<<fname<<"'\n";
			}
		}
	}
	
	enum result_t{
		CUDA_SUCCESS_ = 0,
		CUDA_ERROR_INVALID_VALUE_ = 1,
		CUDA_ERROR_OUT_OF_MEMORY_ =  2,
		CUDA_ERROR_NOT_INITIALIZED_ = 3,
		CUDA_ERROR_DEINITIALIZED_ =  4,
		CUDA_ERROR_PROFILER_DISABLED_ = 5,
		CUDA_ERROR_PROFILER_NOT_INITIALIZED_ = 6,
		CUDA_ERROR_PROFILER_ALREADY_STARTED_ = 7,
		CUDA_ERROR_PROFILER_ALREADY_STOPPED_ = 8,
		CUDA_ERROR_NO_DEVICE_ =  100,
		CUDA_ERROR_INVALID_DEVICE_ = 101,
		CUDA_ERROR_INVALID_IMAGE_ =  200,
		CUDA_ERROR_INVALID_CONTEXT_ = 201,
		CUDA_ERROR_CONTEXT_ALREADY_CURRENT_ = 202,
		CUDA_ERROR_MAP_FAILED_ =  205,
		CUDA_ERROR_UNMAP_FAILED_ =  206,
		CUDA_ERROR_ARRAY_IS_MAPPED_ = 207,
		CUDA_ERROR_ALREADY_MAPPED_ = 208,
		CUDA_ERROR_NO_BINARY_FOR_GPU_ = 209,
		CUDA_ERROR_ALREADY_ACQUIRED_ = 210,
		CUDA_ERROR_NOT_MAPPED_ =  211,
		CUDA_ERROR_NOT_MAPPED_AS_ARRAY_ = 212,
		CUDA_ERROR_NOT_MAPPED_AS_POINTER_ = 213,
		CUDA_ERROR_ECC_UNCORRECTABLE_ = 214,
		CUDA_ERROR_UNSUPPORTED_LIMIT_ = 215,
		CUDA_ERROR_CONTEXT_ALREADY_IN_USE_ = 216,
		CUDA_ERROR_PEER_ACCESS_UNSUPPORTED_ = 217,
		CUDA_ERROR_INVALID_SOURCE_ = 300,
		CUDA_ERROR_FILE_NOT_FOUND_ = 301,
		CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND_ = 302,
		CUDA_ERROR_SHARED_OBJECT_INIT_FAILED_ = 303,
		CUDA_ERROR_OPERATING_SYSTEM_ = 304,
		CUDA_ERROR_INVALID_HANDLE_ = 400,
		CUDA_ERROR_NOT_FOUND_ =  500,
		CUDA_ERROR_NOT_READY_ =  600,
		CUDA_ERROR_LAUNCH_FAILED_ =  700,
		CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES_ = 701,
		CUDA_ERROR_LAUNCH_TIMEOUT_ = 702,
		CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING_ = 703,
		CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED_ = 704,
		CUDA_ERROR_PEER_ACCESS_NOT_ENABLED_ = 705,
		CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE_ = 708,
		CUDA_ERROR_CONTEXT_IS_DESTROYED_ = 709,
		CUDA_ERROR_ASSERT_ =  710,
		CUDA_ERROR_TOO_MANY_PEERS_ = 711,
		CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED_ = 712,
		CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED_ = 713,
		CUDA_ERROR_NOT_PERMITTED_ =  800,
		CUDA_ERROR_NOT_SUPPORTED_ =  801,
		CUDA_ERROR_UNKNOWN_ =  999
	};
	
	enum device_attribute_t {
		CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK_ = 1,
		CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X_ = 2,
		CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y_ = 3,
		CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z_ = 4,
		CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X_ = 5,
		CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y_ = 6,
		CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z_ = 7,
		CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_ = 8,
		CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK_ = 8,
		CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY_ = 9,
		CU_DEVICE_ATTRIBUTE_WARP_SIZE_ = 10,
		CU_DEVICE_ATTRIBUTE_MAX_PITCH_ = 11,
		CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK_ = 12,
		CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK_ = 12,
		CU_DEVICE_ATTRIBUTE_CLOCK_RATE_ = 13,
		CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT_ = 14,
		CU_DEVICE_ATTRIBUTE_GPU_OVERLAP_ = 15,
		CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT_ = 16,
		CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT_ = 17,
		CU_DEVICE_ATTRIBUTE_INTEGRATED_ = 18,
		CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY_ = 19,
		CU_DEVICE_ATTRIBUTE_COMPUTE_MODE_ = 20,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH_ = 21,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH_ = 22,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT_ = 23,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ = 24,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ = 25,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ = 26,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH_ = 27,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT_ = 28,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS_ = 29,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH_ = 27,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT_ = 28,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES_ = 29,
		CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT_ = 30,
		CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS_ = 31,
		CU_DEVICE_ATTRIBUTE_ECC_ENABLED_ = 32,
		CU_DEVICE_ATTRIBUTE_PCI_BUS_ID_ = 33,
		CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID_ = 34,
		CU_DEVICE_ATTRIBUTE_TCC_DRIVER_ = 35,
		CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE_ = 36,
		CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH_ = 37,
		CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE_ = 38,
		CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR_ = 39,
		CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT_ = 40,
		CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING_ = 41,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH_ = 42,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS_ = 43,
		CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER_ = 44,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH_ = 45,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT_ = 46,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE_ = 47,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE_ = 48,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE_ = 49,
		CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID_ = 50,
		CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT_ = 51,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH_ = 52,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH_ = 53,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS_ = 54,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH_ = 55,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH_ = 56,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT_ = 57,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH_ = 58,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT_ = 59,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH_ = 60,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH_ = 61,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS_ = 62,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH_ = 63,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT_ = 64,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS_ = 65,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH_ = 66,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH_ = 67,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS_ = 68,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH_ = 69,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH_ = 70,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT_ = 71,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH_ = 72,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH_ = 73,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT_ = 74,
		CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR_ = 75,     
		CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR_ = 76,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH_ = 77,
		CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED_ = 78,
		CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
		CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
		CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
		CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
		CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
		CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
		CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
		CU_DEVICE_ATTRIBUTE_MAX_
	};
	
	typedef int device_t;
	typedef unsigned int * device_memptr_t;
	typedef struct CUctx_st * context_t;
	typedef struct CUmod_st * module_t;
	typedef struct CUfunc_st * function_t;
	typedef struct CUevent_st * event_t;
	typedef struct CUstream_st * stream_t;
	
	/* initialization */
	std::function<result_t(int)> cuInit;
	/* error handling */
	std::function<result_t(result_t,const char **)> cuGetErrorName;
	std::function<result_t(result_t,const char **)> cuGetErrorString;
	/* version info */
	std::function<result_t(int*)> cuDriverGetVersion;
	/* device management */
	std::function<result_t(device_t*,int)> cuDeviceGet;
	std::function<result_t(int*,device_attribute_t,device_t)> cuDeviceGetAttribute;
	std::function<result_t(int *)> cuDeviceGetCount;
	std::function<result_t(char*,int,device_t)> cuDeviceGetName;
	std::function<result_t(size_t*,device_t)> cuDeviceTotalMem;
	/* context management */
	std::function<result_t(context_t*,device_t)> cuDevicePrimaryCtxRetain;
	std::function<result_t(device_t,unsigned int *, int *)> cuDevicePrimaryCtxGetState;
	std::function<result_t(context_t*,unsigned int,device_t)> cuCtxCreate;
	std::function<result_t(context_t)> cuCtxDestroy;
	std::function<result_t(context_t*)> cuCtxGetCurrent;
	std::function<result_t(context_t)> cuCtxSetCurrent;
	/* module management */
	std::function<result_t(module_t*,const char *)> cuModuleLoad;
	std::function<result_t(function_t*,module_t,const char*)> cuModuleGetFunction;
	std::function<result_t(module_t)> cuModuleUnload;
	/* memory management */
	std::function<result_t(device_memptr_t*,size_t)> cuMemAlloc;
	std::function<result_t(device_memptr_t)> cuMemFree;
	std::function<result_t(device_memptr_t, const void *, size_t)> cuMemcpyHtoD;
	std::function<result_t(void *, device_memptr_t, size_t)> cuMemcpyDtoH;
	/* execution control */
	std::function<result_t(function_t,
						 unsigned int, unsigned int, unsigned int, 
					     unsigned int, unsigned int, unsigned int,
					     unsigned int, 
					     stream_t, 
					     void **,
					     void **)> cuLaunchKernel;

	void addSearchPath(const std::string& path){
		priv::libcu_searchpath.insert(path);
	}

	result_t init(){
		char path_buff[1024];
		#ifdef __linux__
		priv::libcu_searchpath.insert("/usr/lib/");
		priv::libcu_searchpath.insert("/usr/lib32/");
		priv::libcu_searchpath.insert("/usr/lib64/");
		priv::libcu_searchpath.insert("/usr/lib/i386-linux-gnu/");
		priv::libcu_searchpath.insert("/usr/lib/x86_64-linux-gnu/");
		#elif defined __WIN32
        priv::libcu_searchpath.insert("C:/Windows/System32/");
		#endif
		if (priv::locate_cuda_rt(path_buff,sizeof(path_buff)) == 0){
			if(void * p = priv::dp_load(path_buff)){
				priv::libcu_handle = p;
				#define CU_LD(name) priv::load_func(p, name , #name);
				CU_LD(cuInit)
				CU_LD(cuGetErrorString)
				CU_LD(cuGetErrorName)
				CU_LD(cuDriverGetVersion)
				CU_LD(cuDeviceGet)
				CU_LD(cuDeviceGetAttribute)
				CU_LD(cuDeviceGetCount)
				CU_LD(cuDeviceGetName)
				CU_LD(cuDeviceTotalMem)
				
				CU_LD(cuDevicePrimaryCtxRetain)
				CU_LD(cuDevicePrimaryCtxGetState)
				CU_LD(cuCtxCreate)
				CU_LD(cuCtxDestroy)
				CU_LD(cuCtxGetCurrent)
				CU_LD(cuCtxSetCurrent)
				
				CU_LD(cuModuleLoad)
				CU_LD(cuModuleGetFunction)
				CU_LD(cuModuleUnload)
				
				CU_LD(cuMemAlloc)
				CU_LD(cuMemcpyHtoD)
				CU_LD(cuMemcpyDtoH)
				CU_LD(cuMemFree)
				
				CU_LD(cuLaunchKernel)
				#undef CU_LD
				return cuInit(0);
			} else{
				std::cerr << "can't load " << path_buff << ", err: " << priv::dp_error() << "\n";
			}
		}
		return CUDA_ERROR_NOT_INITIALIZED_;
	}
	bool isInitialized(){
		return priv::libcu_handle != 0;
	}
	void cleanup(){
		if (isInitialized()){
			priv::dp_close(priv::libcu_handle);
			priv::libcu_handle=0;
		}
	}
	
	template <typename T>
	class DevicePtr{
	public:
	
		typedef T value_type;
	
		DevicePtr(const void * init_value=0, size_t n=1)
			:devPtr_(0)
		{
			this->realloc(n*sizeof(T));
			if (devPtr_ && init_value){
				cuwr::cuMemcpyHtoD(devPtr_, init_value, size_bytes_);
			}
		}
        DevicePtr(const T& value)
            :DevicePtr((const void*)&value,1)
        {
        }
        DevicePtr(DevicePtr&& other)
            :devPtr_(other.devPtr_)
            ,size_bytes_(other.size_bytes_)
        {
            other.devPtr_=0;
        }

		~DevicePtr(){
			if (devPtr_)
				cuwr::cuMemFree(devPtr_);
		}
		cuwr::result_t realloc(size_t nbytes){
			if (devPtr_){
				cuwr::cuMemFree(devPtr_);
			}
			const cuwr::result_t err = cuwr::cuMemAlloc(&devPtr_, nbytes);
			if( err == 0){
				size_bytes_ = nbytes;
			} else{
				devPtr_ = 0;
				size_bytes_ = 0;
			}
			return err;
		}
		cuwr::result_t load( const void * value ){
			return cuwr::cuMemcpyHtoD(devPtr_,value,size_bytes_);
		}
        bool operator << (const T& var){
            return this->operator <<((const void *)&var);
        }
        bool operator << (const void * value){
            return load(value) == 0;
        }
        DevicePtr& operator = (const T& value){
            this->load((const void *)&value);
            return *this;
        }
        cuwr::result_t store( void * out_buff ){
            return cuwr::cuMemcpyDtoH(out_buff,devPtr_,size_bytes_);
        }
        bool operator >> (T& var){
            return this->operator >>((void *)&var);
        }
        bool operator >> (void * out_buff){
            return store(out_buff) == 0;
        }

		operator T*(){
			return devPtr_;
		}
		T ** ptrAddr(){
			return &devPtr_;
		}
	private:
		T * devPtr_;
		size_t size_bytes_;
	};
	
	class KernelLaunchParams{
		CUWR_NOCPY(KernelLaunchParams)
	public:
		KernelLaunchParams() = default;
		void setGridSize(unsigned int x, unsigned int y, unsigned int z=1){
			gridDimX_ = x;
			gridDimY_ = y;
			gridDimZ_ = z;
		}
		void setBlockSize(unsigned int x, unsigned int y, unsigned int z=1){
			blockDimX_ = x;
			blockDimY_ = y;
			blockDimZ_ = z;
		}
		template <typename T>
		void push(DevicePtr<T>& ptr){
			params_.push_back(ptr.ptrAddr());
		}
		
		void clear(){
			params_.clear();
			extra_.clear();
			stream_=0;
			sharedMemBytes_=0;
			gridDimX_ = gridDimY_ = gridDimZ_ = 1;
			blockDimX_ = blockDimY_ = blockDimZ_ = 1;
		}
	private:
		unsigned int gridDimX_=1, gridDimY_=1, gridDimZ_=1;
		unsigned int blockDimX_=1, blockDimY_=1, blockDimZ_=1;
		unsigned int sharedMemBytes_=0;
		cuwr::stream_t stream_=0;
		std::vector<void *> params_;
		std::vector<void *> extra_;
		friend cuwr::result_t launch_kernel(cuwr::function_t, const KernelLaunchParams&);
	};
	
	cuwr::result_t launch_kernel(cuwr::function_t fn, const KernelLaunchParams& params){
		return cuwr::cuLaunchKernel(fn,params.gridDimX_,params.gridDimY_,params.gridDimZ_,
									   params.blockDimX_,params.blockDimY_,params.blockDimZ_,
									params.sharedMemBytes_,
									params.stream_,
									params.params_.empty() ? 0 : (void**)&params.params_[0],
									params.extra_.empty() ? 0 : (void**)&params.extra_[0]);
	}
	
	class Module{
		CUWR_NOCPY(Module)
	public:
		Module(const char * fname=0)
			:module_(0)
		{
			if (fname){
				load(fname);
			}
		}
		~Module(){
			if (module_)
				cuwr::cuModuleUnload(module_);
		}
		bool isLoaded() const{
			return module_ != 0;
		}
		cuwr::result_t load(const char * fname){
			return cuwr::cuModuleLoad(&module_,fname);
		}
		cuwr::function_t function(const char * name){
			cuwr::function_t fn = 0;
			const cuwr::result_t e = cuwr::cuModuleGetFunction(&fn,module_,name);
			if (e != 0)
				fn=0;
			return fn;
		}
	private:
		cuwr::module_t module_;
	};	
	
	/*!
	 * \brief Wrapper for GPU functionality
	 * */
	class Gpu{
		CUWR_NOCPY(Gpu)
	public:
		Gpu(int dev_number=0)
			:devId_(-1)
			,context_(nullptr)
		{
			if( cuwr::cuDeviceGet(&devId_,0) == 0 ){
				cuwr::cuCtxCreate(&context_,0,devId_);
			}
		}
		~Gpu(){
			if (context_)
				cuwr::cuCtxDestroy(context_);
		}
		/*!
		 * \brief makes this gpu context current for the calling thread
		 * if this gpu context is already current, the call has no effect
		 * */
		void makeCurrent(){
			cuwr::context_t curr=0;
			if(context_ && cuwr::cuCtxGetCurrent(&curr)==0){
				if (curr != context_){
					cuwr::cuCtxSetCurrent(context_);
				}
			}
		}
		const std::string name() const{
			char buff[256];
			if (cuwr::cuDeviceGetName(buff,256,devId_) == 0){
				return std::string(buff);
			}
			return std::string();
		}
		const std::string computeCapabilityStr() const{
			if (context_){
				std::stringstream ss;
				ss << this->attribute(cuwr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR_)
				   << '.'
				   << this->attribute(cuwr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR_);
				return ss.str();
			} else{
				return std::string();
			}
		}
		const size_t totalMemory() const{
			size_t result=0;
			cuwr::cuDeviceTotalMem(&result,devId_);
			return result;
		}
		int attribute(cuwr::device_attribute_t attr) const{
			int result=0;
			cuwr::cuDeviceGetAttribute(&result,attr,devId_);
			return result;
		}
		bool canMapHost() const{
			return this->attribute(cuwr::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY_);
		}	
	private:
		int devId_;
		cuwr::context_t context_;
	};
	
}

#endif
