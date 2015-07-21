#include "cuwrap.h"
#include <cstdio>
#include <iostream>
#ifdef _WIN32
	#include <windows.h>
#elif defined __linux__
    #include <dlfcn.h>
#else
	#error Unsupported platform !
#endif
#include <set>


namespace cuwr{
	
	namespace priv{
		#ifdef _WIN32
			typedef HMODULE dplibhandle_priv_t_;
		#elif defined __linux__
			typedef void * dplibhandle_priv_t_;
		#endif
		
		static std::set<std::string> libcu_searchpath;
		static dplibhandle_priv_t_ libcu_handle;
		
        inline void * dp_load(const char * path){
			#ifdef _WIN32
				return LoadLibraryA(path);
			#elif defined __linux__
				dlerror();
				return dlopen(path,RTLD_LAZY | RTLD_LOCAL);
			#endif
		}

        inline void * dp_symbol(void * h, const char * sym){
			dplibhandle_priv_t_ p = (dplibhandle_priv_t_)h;
			#ifdef _WIN32
				return (void *)GetProcAddress(p,sym);
			#elif defined __linux__
				dlerror();
				return dlsym(p,sym);
			#endif
		}

        inline const char * dp_error(){
			#ifdef _WIN32
				static char buff[256];
				snprintf(buff,256,"ERROR CODE: %lu\n",GetLastError());
				return buff;
			#elif defined __linux__
				return dlerror();
			#endif
		}

        inline void dp_close(void * h){
			dplibhandle_priv_t_ p = (dplibhandle_priv_t_)h;
			if (p){
				#ifdef _WIN32
					FreeLibrary(p);
				#elif defined __linux__
					dlclose(p);
				#endif
			}
		}
		
        inline int locate_cuda_rt(char * out_path, int maxLen){
			#ifdef __linux__
				const std::string lname = "libcuda.so";
			#elif defined __WIN32
				const std::string lname = "nvcuda.dll";
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
			#ifdef __linux__
			#define CALLCONV
			#elif defined __WIN32
			#define CALLCONV __stdcall
			#endif
			fc = (R (CALLCONV *)(Arg...))priv::dp_symbol(lib_handle,fname);
			if (!fc){
				std::cerr << "could not locate symbol '"<<fname<<"'\n";
			}
		}
	}
	
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
    /* occupancy */
    std::function<result_t(int*, function_t, int, size_t)> cuOccupancyMaxActiveBlocksPerMultiprocessor;
    std::function<result_t(int *, int *, function_t, function_t,size_t,int)> cuOccupancyMaxPotentialBlockSize;

    /* streams */
    std::function<result_t(event_t*,unsigned int)> cuEventCreate;
    std::function<result_t(event_t)> cuEventDestroy;
    std::function<result_t(float *,event_t,event_t)> cuEventElapsedTime;
    std::function<result_t(event_t)> cuEventQuery;
    std::function<result_t(event_t,stream_t)> cuEventRecord;
    std::function<result_t(event_t)> cuEventSynchronize;



    void addSearchPath(const std::string& path){
		priv::libcu_searchpath.insert(path);
	}
	
	std::string tostr(const result_t errCode){
		std::string result;
		const char * ptr = nullptr;
		cuwr::cuGetErrorName(errCode,&ptr);
		if (ptr){
			result = std::string(ptr);
		} else{
			result = "[ERR_UNKNOWN]";
		}
		result += ": ";
		ptr = nullptr;
		cuwr::cuGetErrorString(errCode,&ptr);
		if (ptr){
			result += std::string(ptr);
		} else{
			result += "(no description)";
		}
		result += '\n';
		return result;
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
		priv::libcu_searchpath.insert("C:/Windows/SysWOW64/");
        priv::libcu_searchpath.insert("C:/Windows/System32/");
        #endif
        if (priv::locate_cuda_rt(path_buff,sizeof(path_buff)) == 0){
			if(void * p = priv::dp_load(path_buff)){
				priv::libcu_handle = (priv::dplibhandle_priv_t_)p;
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
                CU_LD(cuOccupancyMaxPotentialBlockSize)
                CU_LD(cuOccupancyMaxActiveBlocksPerMultiprocessor)

                CU_LD(cuEventCreate)
                CU_LD(cuEventDestroy)
                CU_LD(cuEventElapsedTime)
                CU_LD(cuEventQuery)
                CU_LD(cuEventRecord)
                CU_LD(cuEventSynchronize)
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
	
    cuwr::result_t launch_kernel(cuwr::function_t fn, const KernelLaunchParams& params){
		return cuwr::cuLaunchKernel(fn,params.gridDimX_,params.gridDimY_,params.gridDimZ_,
									   params.blockDimX_,params.blockDimY_,params.blockDimZ_,
									params.sharedMemBytes_,
									params.stream_,
									params.params_.empty() ? 0 : (void**)&params.params_[0],
									params.extra_.empty() ? 0 : (void**)&params.extra_[0]);
	}
	
}
