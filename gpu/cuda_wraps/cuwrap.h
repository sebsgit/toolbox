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
//TODO memory
//TODO module
//TODO autodetect best kernel launch size

#include <vector>
#include <functional>
#include <string>
#include <sstream>
#include <cstring>

#define CUWR_NOCPY(K) private:						\
                        K (const K&) = delete;      \
                        K(K &&) = delete;           \
                        K& operator = (const K&) = delete;	\
                        K& operator = (K&&) = delete;

namespace cuwr{
	
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
		CUDA_ERROR_ILLEGAL_ADDRESS_ =  700,
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

    enum event_flags_t{
        CU_EVENT_DEFAULT_ = 0x0,
        CU_EVENT_BLOCKING_SYNC_ = 0x1,
        CU_EVENT_DISABLE_TIMING_ = 0x2,
        CU_EVENT_INTERPROCESS_ = 0x4
    };

    enum host_mem_alloc_flags_t{
        CU_MEMHOSTALLOC_PORTABLE_ = 0x01,
        CU_MEMHOSTALLOC_DEVICEMAP_ = 0x02,
        CU_MEMHOSTALLOC_WRITECOMBINED_ = 0x04
    };

    enum host_mem_register_flags_t{
        CU_MEMHOSTREGISTER_PORTABLE_  = 0x01,
        CU_MEMHOSTREGISTER_DEVICEMAP_ = 0x02
    };

    enum stream_creation_flags_t{
        CU_STREAM_DEFAULT = 0x0,
        CU_STREAM_NON_BLOCKING = 0x1
    };
	
	typedef int device_t;
	typedef unsigned int * device_memptr_t; /*unsigned integer type whose size matches the size of a pointer*/
	typedef struct CUctx_st * context_t;
	typedef struct CUmod_st * module_t;
	typedef struct CUfunc_st * function_t;
	typedef struct CUevent_st * event_t;
	typedef struct CUstream_st * stream_t;
	
	/* initialization */
	extern std::function<result_t(int)> cuInit;
	/* error handling */
	extern std::function<result_t(result_t,const char **)> cuGetErrorName;
	extern std::function<result_t(result_t,const char **)> cuGetErrorString;
	/* version info */
	extern std::function<result_t(int*)> cuDriverGetVersion;
	/* device management */
	extern std::function<result_t(device_t*,int)> cuDeviceGet;
	extern std::function<result_t(int*,device_attribute_t,device_t)> cuDeviceGetAttribute;
	extern std::function<result_t(int *)> cuDeviceGetCount;
	extern std::function<result_t(char*,int,device_t)> cuDeviceGetName;
	extern std::function<result_t(size_t*,device_t)> cuDeviceTotalMem;
	/* context management */
	extern std::function<result_t(context_t*,device_t)> cuDevicePrimaryCtxRetain;
	extern std::function<result_t(device_t,unsigned int *, int *)> cuDevicePrimaryCtxGetState;
	extern std::function<result_t(context_t*,unsigned int,device_t)> cuCtxCreate;
	extern std::function<result_t(context_t)> cuCtxDestroy;
	extern std::function<result_t(context_t*)> cuCtxGetCurrent;
	extern std::function<result_t(context_t)> cuCtxSetCurrent;
    extern std::function<result_t(void)> cuCtxSynchronize;
	/* module management */
	extern std::function<result_t(module_t*,const char *)> cuModuleLoad;
	extern std::function<result_t(function_t*,module_t,const char*)> cuModuleGetFunction;
	extern std::function<result_t(module_t)> cuModuleUnload;
    /* memory management */
    extern std::function<result_t(size_t*,size_t*)> cuMemGetInfo;
	extern std::function<result_t(device_memptr_t*,size_t)> cuMemAlloc;
	extern std::function<result_t(device_memptr_t)> cuMemFree;
    extern std::function<result_t(void **,size_t)> cuMemAllocHost;
    extern std::function<result_t(void **, size_t, unsigned int)> cuMemHostAlloc;
    extern std::function<result_t(device_memptr_t*,void *,unsigned int)> cuMemHostGetDevicePointer;
    extern std::function<result_t(void*,size_t,unsigned int)> cuMemHostRegister;
    extern std::function<result_t(void*)> cuMemHostUnregister;
    extern std::function<result_t(void *)> cuMemFreeHost;
    extern std::function<result_t(device_memptr_t,device_memptr_t,size_t)> cuMemcpy;
	extern std::function<result_t(device_memptr_t, const void *, size_t)> cuMemcpyHtoD;
	extern std::function<result_t(void *, device_memptr_t, size_t)> cuMemcpyDtoH;
    extern std::function<result_t(void *, device_memptr_t, size_t)> cuMemcpyDtoD;

	/* stream management */
    typedef void (* stream_callback_t )( stream_t, result_t, void*);
    extern std::function<result_t(stream_t,stream_callback_t,void*,unsigned int)> cuStreamAddCallback;
    extern std::function<result_t(stream_t*, unsigned int)> cuStreamCreate;
    extern std::function<result_t(stream_t*, unsigned int, int)> cuStreamCreateWithPriority;
    extern std::function<result_t(stream_t)> cuStreamDestroy;
    extern std::function<result_t(stream_t, unsigned int *)> cuStreamGetFlags;
    extern std::function<result_t(stream_t, int *)> cuStreamGetPriority;
    extern std::function<result_t(stream_t)> cuStreamQuery;
    extern std::function<result_t(stream_t)> cuStreamSynchronize;
    extern std::function<result_t(stream_t,event_t,unsigned int)> cuStreamWaitEvent;

	/* execution control */
	extern std::function<result_t(function_t,
						 unsigned int, unsigned int, unsigned int, 
					     unsigned int, unsigned int, unsigned int,
					     unsigned int, 
					     stream_t, 
					     void **,
					     void **)> cuLaunchKernel;
    /* occupancy */
    extern std::function<result_t(int*, function_t, int, size_t)> cuOccupancyMaxActiveBlocksPerMultiprocessor;
    extern std::function<result_t(int *, int *, function_t, function_t,size_t,int)> cuOccupancyMaxPotentialBlockSize;

    /* events */
    extern std::function<result_t(event_t*,unsigned int)> cuEventCreate;
    extern std::function<result_t(event_t)> cuEventDestroy;
    extern std::function<result_t(float *,event_t,event_t)> cuEventElapsedTime;
    extern std::function<result_t(event_t)> cuEventQuery;
    extern std::function<result_t(event_t,stream_t)> cuEventRecord;
    extern std::function<result_t(event_t)> cuEventSynchronize;

    extern void addSearchPath(const std::string& path);

	class Gpu;

    extern result_t init();
    extern bool isInitialized();
    extern cuwr::Gpu& defaultGpu();
    extern void cleanup();
    
    extern std::string tostr(const result_t errCode);
	
    class Exception : public std::exception{
    public:
        Exception(cuwr::result_t errCode)
            :err_(errCode)
            ,buff_(cuwr::tostr(errCode))
        {
            
        }
        const char * what() const 
						#ifdef __linux__
							noexcept(true) 
						#endif	
							override
		{
            return buff_.c_str();
        }
    private:
        const cuwr::result_t err_;
        const std::string buff_;
    };

    /*!
     * \brief base class for device values
     */
    class DeviceValueBase{
    public:
        virtual ~DeviceValueBase(){
        }
        virtual cuwr::device_memptr_t * ptrAddress() const = 0;
        virtual const void * hostAddress() const = 0;
        virtual size_t size() const = 0;
    };

    class DeviceMemAllocator{
    public:
        typedef cuwr::device_memptr_t pointer_type;

        static void zero(pointer_type * p){
            *p = 0;
        }
        static bool isNull(const pointer_type& p){
            return p==0;
        }
        static cuwr::result_t alloc(pointer_type * outp, const size_t nBytes){
            return cuwr::cuMemAlloc(outp,nBytes);
        }
        static cuwr::result_t free(pointer_type ptr){
            return cuwr::cuMemFree(ptr);
        }
        static cuwr::result_t copyToDevice(pointer_type dest, const void * src, const size_t nbytes){
            return cuwr::cuMemcpyHtoD(dest,src,nbytes);
        }
        static cuwr::result_t copyToHost(void * dest, pointer_type src, const size_t nbytes){
            return cuwr::cuMemcpyDtoH(dest,src,nbytes);
        }
        static cuwr::result_t copyDeviceToDevice(pointer_type dest, const pointer_type& src, const size_t nbytes){
            return cuwr::cuMemcpyDtoD(dest,src,nbytes);
        }
        static cuwr::device_memptr_t * deviceAddress(const pointer_type & src){
            return (cuwr::device_memptr_t*)&src;
        }
        static const void * hostAddress(const pointer_type& /*src*/){
            throw Exception(cuwr::CUDA_ERROR_NOT_MAPPED_);
            return nullptr;
        }
    };
    class DeviceMemPinnedAllocator{
    public:
        typedef struct{
            void * hostp_ = 0;
            cuwr::device_memptr_t devp_ = 0;
        } pointer_type;

        static void zero(pointer_type * p){
            p->devp_ = 0;
            p->hostp_ = 0;
        }
        static bool isNull(const pointer_type& p){
            return p.hostp_==0;
        }
        static cuwr::result_t alloc(pointer_type * outp, const size_t nBytes){
            cuwr::result_t err = cuwr::cuMemHostAlloc(&outp->hostp_,nBytes,cuwr::CU_MEMHOSTALLOC_DEVICEMAP_);
            if (err == 0){
                err = cuwr::cuMemHostGetDevicePointer(&outp->devp_,outp->hostp_,0);
            }
            return err;
        }
        static cuwr::result_t free(pointer_type ptr){
            return cuwr::cuMemFreeHost(ptr.hostp_);
        }
        static cuwr::result_t copyToDevice(pointer_type dest, const void * src, const size_t nbytes){
            return memcpy(dest.hostp_,src,nbytes) != 0 ? cuwr::CUDA_SUCCESS_ : cuwr::CUDA_ERROR_ILLEGAL_ADDRESS_;
        }
        static cuwr::result_t copyToHost(void * dest, pointer_type src, const size_t nbytes){
            const cuwr::result_t err = cuwr::cuCtxSynchronize();
            if (err == 0)
                memcpy(dest,src.hostp_,nbytes);
            return err;
        }
        static cuwr::result_t copyDeviceToDevice(pointer_type dest, const pointer_type& src, const size_t nbytes){
            return memcpy(dest.hostp_,src.hostp_,nbytes) != 0 ? cuwr::CUDA_SUCCESS_ : cuwr::CUDA_ERROR_ILLEGAL_ADDRESS_;
        }
        static cuwr::device_memptr_t * deviceAddress(const pointer_type & src){
            return (cuwr::device_memptr_t*)&src.devp_;
        }
        static const void * hostAddress(const pointer_type& src){
            return src.hostp_;
        }
    };
    typedef DeviceMemAllocator DefaultAllocator;

    template <typename T, typename Alloc = DefaultAllocator>
    class DeviceValue : public DeviceValueBase{
	public:
	
		typedef T value_type;
        typedef Alloc allocator_type;
	
        DeviceValue(const void * init_value=0)
		{
            Alloc::zero(&devPtr_);
            cuwr::result_t err = Alloc::alloc(&devPtr_, sizeof(T));
            if( err == 0){
                if (init_value)
                    err = Alloc::copyToDevice(devPtr_, init_value, sizeof(T));
            }
            if (err != 0){
                throw cuwr::Exception(err);
            }
		}
        DeviceValue(const T& value)
            :DeviceValue((const void*)&value)
        {
        }
        DeviceValue(const DeviceValue& other){
            Alloc::zero(&devPtr_);
            cuwr::result_t err = Alloc::alloc(&devPtr_, sizeof(T));
            if( err == 0){
                err = Alloc::copyDeviceToDevice(devPtr_,other.devPtr_,sizeof(T));
            }
            if (err != 0){
                throw cuwr::Exception(err);
            }
        }
        DeviceValue(DeviceValue&& other)
            :devPtr_(other.devPtr_)
        {
            Alloc::zero(&other.devPtr_);
        }
        DeviceValue& operator = (DeviceValue&& other){
            devPtr_ = other.devPtr_;
            Alloc::zero(&other.devPtr_);
            return *this;
        }
        DeviceValue& operator = (const DeviceValue& other){
            if (Alloc::isNull(devPtr_)==false){
                Alloc::free(devPtr_);
                Alloc::zero(&devPtr_);
            }
            cuwr::result_t err = Alloc::alloc(&devPtr_, sizeof(T));
            if( err == 0){
                err = Alloc::copyDeviceToDevice(devPtr_,other.devPtr_,sizeof(T));
            }
            if (err != 0){
                throw cuwr::Exception(err);
            }
            return *this;
        }
        ~DeviceValue(){
            if (Alloc::isNull(devPtr_)==false)
                Alloc::free(devPtr_);
		}
        void clear(){
            if (Alloc::isNull(devPtr_)==false){
                Alloc::free(devPtr_);
                Alloc::zero(&devPtr_);
            }
        }
		cuwr::result_t load( const void * value ){
            return Alloc::copyToDevice(devPtr_,value,sizeof(T));
		}
        bool operator << (const T& var){
            return this->operator <<((const void *)&var);
        }
        bool operator << (const void * value){
            return load(value) == 0;
        }
        DeviceValue& operator = (const T& value){
            this->load((const void *)&value);
            return *this;
        }
        cuwr::result_t store( void * out_buff ) const{
            return Alloc::copyToHost(out_buff,devPtr_,sizeof(T));
        }
        bool operator >> (T& var) const{
            return this->operator >>((void *)&var);
        }
        bool operator >> (void * out_buff) const{
            return store(out_buff) == 0;
        }

        operator T() const{
            T tmp;
            this->store(&tmp);
            return tmp;
		}
        T * operator -> (){
            return (T*)hostAddress();
        }
        const T * operator -> () const{
            return (const T*)hostAddress();
        }
        device_memptr_t * ptrAddress() const override{
            return Alloc::deviceAddress(devPtr_);
		}
        const void * hostAddress() const{
            return Alloc::hostAddress(devPtr_);
        }
        size_t size() const override{
            return sizeof(T);
		}

	private:
        typename Alloc::pointer_type devPtr_;
	};

    template <typename T, typename Alloc = DefaultAllocator>
    class DeviceArray : public DeviceValueBase{
    public:
        typedef T value_type;
        DeviceArray(const size_t initSize=0)
            :count_(0)
        {
            Alloc::zero(&devPtr_);
            if (initSize > 0)
                this->resize(initSize);
        }
        DeviceArray(DeviceArray&& other){
            devPtr_ = other.devPtr_;
            count_ = other.count_;
            Alloc::zero(&other.devPtr_);
            other.count_ = 0;
        }
        DeviceArray& operator=(DeviceArray&& other){
            devPtr_ = other.devPtr_;
            count_ = other.count_;
            Alloc::zero(&other.devPtr_);
            other.count_ = 0;
            return *this;
        }
        DeviceArray(const DeviceArray& other)
        {
            Alloc::zero(&devPtr_);
            this->resize(other.count_);
            Alloc::copyDeviceToDevice(devPtr_,other.devPtr_,sizeof(T)*count_);
        }
        DeviceArray& operator = (const DeviceArray& other)
        {
            Alloc::zero(&devPtr_);
            this->resize(other.count_);
            Alloc::copyDeviceToDevice(devPtr_,other.devPtr_,sizeof(T)*count_);
            return *this;
        }

        ~DeviceArray(){
            if (Alloc::isNull(devPtr_)==false){
                Alloc::free(devPtr_);
            }
        }

        cuwr::result_t resize(const size_t count){
            if (Alloc::isNull(devPtr_)==false){
                Alloc::free(devPtr_);
                Alloc::zero(&devPtr_);
            }
            const cuwr::result_t errCode = Alloc::alloc(&devPtr_,count*sizeof(T));
            if (errCode==0){
                this->count_ = count;
            }
            return errCode;
        }
        cuwr::result_t resize(const size_t count, const T& initValue){
            const cuwr::result_t errCode = this->resize(count);
            if (errCode==0){
                this->init(initValue);
            }
            return errCode;
        }
        cuwr::result_t load(const void * value, const size_t count = 0){
            return Alloc::copyToDevice(devPtr_,value,sizeof(T)*(count > 0 ? count : this->count_));
        }
        cuwr::result_t store(void * out, const size_t count = 0) const{
            return Alloc::copyToHost(out,devPtr_,sizeof(T)*(count > 0 ? count : this->count_));
        }
        void init(const T& value){
            const std::vector<T> buffer(this->count(),value);
            this->load(&buffer[0]);
        }

        size_t size() const override{
            return this->count_*sizeof(T);
        }
        size_t count() const{
            return this->count_;
        }
        std::vector<T> to_vector() const{
            std::vector<T> result;
            result.resize(this->count());
            this->store(&result[0]);
            return result;
        }
        device_memptr_t * ptrAddress() const override{
            return Alloc::deviceAddress(devPtr_);
        }
        const void * hostAddress() const{
            return Alloc::hostAddress(devPtr_);
        }
        typename Alloc::pointer_type dataPtr() const{
            return this->devPtr_;
        }

    private:
        typename Alloc::pointer_type devPtr_;
        size_t count_;
    };
	
	class KernelLaunchParams{
		CUWR_NOCPY(KernelLaunchParams)
	public:
        KernelLaunchParams() = default;
        unsigned int gridX() const{
            return this->gridDimX_;
        }
        unsigned int gridY() const{
            return this->gridDimY_;
        }
        unsigned int blockX() const{
            return this->blockDimX_;
        }
        unsigned int blockY() const{
            return this->blockDimY_;
        }
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
        void setStream(cuwr::stream_t stream){
            this->stream_ = stream;
        }
        void setSharedMemoryCount(unsigned int numBytes){
            this->sharedMemBytes_ = numBytes;
        }
        void push(const DeviceValueBase& ptr){
            params_.push_back(ptr.ptrAddress());
		}
        template <typename T>
        void push(const T * ptr){
            params_.push_back((void *)ptr);
        }
		
        /*
         finds the smallest 2d rectangle with area >= n_elems
         both width and height of the rectangle must be powers of 2
         */
        static void find_2d_box(const size_t n_elems, size_t * w, size_t * h){
            size_t p2 = 1;
            while (p2*p2 < n_elems){
                p2 *= 2;
            }
            *w = p2;
            size_t p2_h = p2/2;
            while (p2*p2_h > n_elems){
                p2_h /= 2;
            }
            if (p2_h == 0)
                p2_h = 1;
            *h = (p2_h*p2 > n_elems) ? p2_h :p2_h*2;
        }

        /* autodetect launch parameters for per-element processing
           - one thread processes one element
           //TODO
           */
        void autodetect(const size_t count, const size_t blockSize=32){
            const size_t blockWidth = blockSize;
            const size_t blockHeight = blockSize;
            const size_t threadsInBlock = blockWidth*blockHeight;
            const size_t blocksNeeded = (count/threadsInBlock)+1;
            size_t gridW, gridH;
            find_2d_box(blocksNeeded,&gridW,&gridH);
            this->setBlockSize((unsigned int)blockWidth,(unsigned int)blockHeight);
			this->setGridSize((unsigned int)gridW, (unsigned int)gridH);
        }
        /* autodetect launch grid size that covers the [width x height] rectangle */
        void autodetect(const std::pair<size_t,size_t>& size, const size_t blockSize=32){
            const size_t blockWidth = blockSize;
            const size_t blockHeight = blockSize;
            const size_t blocksNeededW = (size.first/blockWidth)+1;
            const size_t blocksNeededH = (size.second/blockHeight)+1;
			this->setBlockSize((unsigned int)blockWidth, (unsigned int)blockHeight);
			this->setGridSize((unsigned int)blocksNeededW, (unsigned int)blocksNeededH);
        }

		void clear(){
			params_.clear();
			extra_.clear();
			stream_=0;
			sharedMemBytes_=0;
			gridDimX_ = gridDimY_ = gridDimZ_ = 1;
			blockDimX_ = blockDimY_ = blockDimZ_ = 1;
		}
        void clearParameters(){
            this->params_.clear();
            this->extra_.clear();
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
		cuwr::function_t function(const char * name, cuwr::result_t * errCode = nullptr){
			cuwr::function_t fn = 0;
			const cuwr::result_t err = cuwr::cuModuleGetFunction(&fn,module_,name);
			if (err != 0)
				fn=0;
			if (errCode){
				*errCode = err;
			}
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
            if( cuwr::cuDeviceGet(&devId_,dev_number) == 0 ){
				cuwr::cuCtxCreate(&context_,0,devId_);
			}
		}
		~Gpu(){
			if (context_)
				cuwr::cuCtxDestroy(context_);
		}
		bool isInitialized() const{
			return context_ != nullptr;
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
        size_t totalMemory() const{
			size_t result=0;
			cuwr::cuDeviceTotalMem(&result,devId_);
			return result;
		}
		int attribute(cuwr::device_attribute_t attr) const{
			int result=0;
			cuwr::cuDeviceGetAttribute(&result,attr,devId_);
			return result;
		}
	private:
		int devId_;
		cuwr::context_t context_;
	};

    /*!
     * \brief event timer
     *     Timer t;
     *     t.start();
     *      ...
     *     t.stop();
     *     float msec = t.elapsed();
     */
    class Timer{
    public:
        Timer(unsigned int flags = cuwr::CU_EVENT_DEFAULT_)
            :result_(0.0f)
            ,startOk_(false)
            ,stopOk_(false)
        {
            this->startOk_ = (cuwr::cuEventCreate(&start_,flags)==0);
            this->stopOk_ = (cuwr::cuEventCreate(&stop_,flags) == 0);
        }
        ~Timer(){
            if (this->startOk_)
                cuwr::cuEventDestroy(start_);
            if (this->stopOk_)
                cuwr::cuEventDestroy(stop_);
        }
        bool start(stream_t stream=0){
            return cuwr::cuEventRecord(start_,stream)==0;
        }
        bool stop(stream_t stream=0){
            return cuwr::cuEventRecord(stop_,stream)==0;
        }
        float elapsed(){
            float result=0.0f;
            cuwr::cuEventSynchronize(stop_);
            cuwr::cuEventElapsedTime(&result,start_,stop_);
            return result;
        }

    private:
        cuwr::event_t start_;
        cuwr::event_t stop_;
        float result_;
        bool startOk_;
        bool stopOk_;
    };
	
	extern cuwr::result_t launch_kernel(cuwr::function_t fn, const KernelLaunchParams& params);
	
}

#endif
