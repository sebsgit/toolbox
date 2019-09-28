#pragma once

#include "so_loader.hpp"

#include <CL/cl.h>
#include <array>
#include <sstream>
#include <string>
#include <vector>

#define DECLARE_CL_API(name) inline decltype(::name)*(name) = nullptr
#define LOAD_CL_API(name) name = reinterpret_cast<decltype(name)>(library_handle.symbol(#name))

#define NON_COPYABLE(Class)       \
    Class(const Class&) = delete; \
    Class& operator=(const Class&) = delete

namespace opencl_rt {

inline so_loader::library library_handle;
DECLARE_CL_API(clGetPlatformIDs);
DECLARE_CL_API(clGetPlatformInfo);
DECLARE_CL_API(clGetDeviceIDs);
DECLARE_CL_API(clGetDeviceInfo);
DECLARE_CL_API(clCreateContext);
DECLARE_CL_API(clGetContextInfo);
DECLARE_CL_API(clReleaseContext);
DECLARE_CL_API(clCreateCommandQueue);
DECLARE_CL_API(clFinish);
DECLARE_CL_API(clReleaseCommandQueue);
DECLARE_CL_API(clCreateProgramWithSource);
DECLARE_CL_API(clBuildProgram);
DECLARE_CL_API(clReleaseProgram);
DECLARE_CL_API(clCreateKernel);
DECLARE_CL_API(clSetKernelArg);
DECLARE_CL_API(clReleaseKernel);
DECLARE_CL_API(clCreateBuffer);
DECLARE_CL_API(clEnqueueReadBuffer);
DECLARE_CL_API(clReleaseMemObject);
DECLARE_CL_API(clEnqueueNDRangeKernel);
DECLARE_CL_API(clGetProgramBuildInfo);

inline bool load(const std::string& libraryPath)
{
    library_handle = so_loader::library(libraryPath);
    if (library_handle.is_open()) {
        LOAD_CL_API(clGetPlatformIDs);
        LOAD_CL_API(clGetPlatformInfo);
        LOAD_CL_API(clGetDeviceIDs);
        LOAD_CL_API(clGetDeviceInfo);
        LOAD_CL_API(clCreateContext);
        LOAD_CL_API(clGetContextInfo);
        LOAD_CL_API(clReleaseContext);
        LOAD_CL_API(clCreateCommandQueue);
        LOAD_CL_API(clFinish);
        LOAD_CL_API(clReleaseCommandQueue);
        LOAD_CL_API(clCreateProgramWithSource);
        LOAD_CL_API(clBuildProgram);
        LOAD_CL_API(clReleaseProgram);
        LOAD_CL_API(clCreateKernel);
        LOAD_CL_API(clSetKernelArg);
        LOAD_CL_API(clReleaseKernel);
        LOAD_CL_API(clCreateBuffer);
        LOAD_CL_API(clEnqueueReadBuffer);
        LOAD_CL_API(clReleaseMemObject);
        LOAD_CL_API(clEnqueueNDRangeKernel);
        LOAD_CL_API(clGetProgramBuildInfo);
    }
    return library_handle.is_open();
}
void close()
{
    library_handle.close();
}

template <typename T>
class backend {
public:
    explicit backend(T h) noexcept
        : _handle(h)
    {
    }
    const auto& handle() const noexcept { return _handle; }

protected:
    void set(T value) noexcept { _handle = value; }

private:
    T _handle;
};

class error : public std::runtime_error {
public:
    explicit error(cl_int code, const std::string& extra_info = std::string())
        : std::runtime_error(extra_info + error_message(code))
    {
    }

private:
    static std::string error_message(cl_int code)
    {
        std::stringstream ss;
        ss << "ERROR: " << code;
        return ss.str();
    }
};

#define THROW_ERROR(code) throw error((code), __PRETTY_FUNCTION__)

template <int param>
struct device_param_trait;

template <int param>
struct context_param_trait;

template <>
struct device_param_trait<CL_DEVICE_NAME> {
    using type = std::string;
};
template <>
struct device_param_trait<CL_DRIVER_VERSION> {
    using type = std::string;
};
template <>
struct device_param_trait<CL_DEVICE_VERSION> {
    using type = std::string;
};
template <>
struct device_param_trait<CL_DEVICE_AVAILABLE> {
    using type = cl_bool;
};
template <>
struct device_param_trait<CL_DEVICE_PLATFORM> {
    using type = cl_platform_id;
};

template <>
struct context_param_trait<CL_CONTEXT_DEVICES> {
    using type = std::vector<cl_device_id>;
};

namespace priv {
    template <typename T>
    void resize(T& /*param*/, size_t /*size*/) {}
    template <>
    void resize<std::string>(std::string& param, size_t size)
    {
        param.resize(size);
    }
    //TODO: generic
    template <>
    void resize<std::vector<cl_device_id>>(std::vector<cl_device_id>& param, size_t size)
    {
        param.resize(size);
    }

    template <typename T>
    void* address(T& param) { return &param; }
    template <>
    void* address<std::string>(std::string& param) { return &param[0]; }
    template <>
    void* address<std::vector<cl_device_id>>(std::vector<cl_device_id>& param) { return param.data(); }
}

class device : public backend<cl_device_id> {
public:
    using backend::backend;

    template <int param>
    [[nodiscard]] auto info() const
    {
        size_t size = 0;
        opencl_rt::clGetDeviceInfo(handle(), param, 0, nullptr, &size);
        typename device_param_trait<param>::type result;
        priv::resize(result, size);
        opencl_rt::clGetDeviceInfo(handle(), param, size, priv::address(result), nullptr);
        return result;
    }
};

class platform : public backend<cl_platform_id> {
public:
    using backend::backend;

    [[nodiscard]] std::string info(cl_platform_info param) const
    {
        size_t size = 0;
        opencl_rt::clGetPlatformInfo(handle(), param, 0, nullptr, &size);
        std::string result(size, ' ');
        opencl_rt::clGetPlatformInfo(handle(), param, result.size(), &result[0], nullptr);
        return result;
    }

    [[nodiscard]] std::vector<device> devices(cl_device_type type) const
    {
        cl_uint numDevices = 0;
        opencl_rt::clGetDeviceIDs(handle(), type, 0, nullptr, &numDevices);
        std::vector<cl_device_id> devices(numDevices);
        opencl_rt::clGetDeviceIDs(handle(), type, numDevices, &devices[0], nullptr);
        std::vector<device> result;
        result.reserve(devices.size());
        for (auto d : devices)
            result.emplace_back(d);
        return result;
    }
};

class buffer : public backend<cl_mem> {
public:
    using backend::backend;

    NON_COPYABLE(buffer);

    buffer(buffer&& other) noexcept
        : backend(other.handle())
    {
        other.set(nullptr);
    }
    buffer& operator=(buffer&& other) noexcept
    {
        if (handle() != other.handle()) {
            if (handle())
                opencl_rt::clReleaseMemObject(handle());
            set(other.handle());
            other.set(nullptr);
        }
        return *this;
    }
};

class context : public backend<cl_context> {
public:
    using backend::backend;

    NON_COPYABLE(context);

    explicit context(device& dev)
        : backend(create(dev))
    {
    }
    context(context&& other) noexcept
        : backend(other.handle())
    {
        other.set(nullptr);
    }
    context& operator=(context&& other) noexcept
    {
        if (handle() != other.handle()) {
            if (handle())
                opencl_rt::clReleaseContext(handle());
            set(other.handle());
            other.set(nullptr);
        }
        return *this;
    }
    ~context()
    {
        if (handle())
            opencl_rt::clReleaseContext(handle());
    }

    template <int param>
    auto info() const
    {
        size_t size = 0;
        opencl_rt::clGetContextInfo(handle(), param, 0, nullptr, &size);
        typename context_param_trait<param>::type result;
        priv::resize(result, size);
        opencl_rt::clGetContextInfo(handle(), param, size, priv::address(result), nullptr);
        return result;
    }

    buffer createBuffer(cl_mem_flags flags, size_t size, void* host_ptr)
    {
        cl_int error_code = 0;
        auto result = opencl_rt::clCreateBuffer(handle(), flags, size, host_ptr, &error_code);
        return buffer(result);
    }

private:
    static cl_context create(device& dev)
    {
        std::array<cl_context_properties, 3> properties { CL_CONTEXT_PLATFORM,
            reinterpret_cast<cl_context_properties>(dev.info<CL_DEVICE_PLATFORM>()),
            0 };
        cl_int error_code = 0;
        auto result = opencl_rt::clCreateContext(properties.data(), 1, &dev.handle(), nullptr, nullptr, &error_code);
        if (error_code != CL_SUCCESS)
            THROW_ERROR(error_code);
        return result;
    }
};

class kernel : public backend<cl_kernel> {
public:
    using backend::backend;

    kernel(kernel&& other) noexcept
        : backend(other.handle())
    {
        other.set(nullptr);
    }
    kernel& operator=(kernel&& other) noexcept
    {
        if (handle() != other.handle()) {
            if (handle())
                opencl_rt::clReleaseKernel(handle());
            this->set(other.handle());
            other.set(nullptr);
        }
        return *this;
    }

    ~kernel()
    {
        if (handle())
            opencl_rt::clReleaseKernel(handle());
    }

    template <typename T>
    void setArg(cl_uint index, const T& value)
    {
        auto result = opencl_rt::clSetKernelArg(handle(), index, sizeof(T), &value);
        if (result != CL_SUCCESS)
            THROW_ERROR(result);
    }
};

class command_queue : public backend<cl_command_queue> {
public:
    using backend::backend;

    NON_COPYABLE(command_queue);

    command_queue(context& ctx, device& dev)
        : backend(create(ctx, dev))
    {
    }
    command_queue(command_queue&& other) noexcept
        : backend(other.handle())
    {
        other.set(nullptr);
    }
    command_queue& operator=(command_queue&& other) noexcept
    {
        if (handle() != other.handle()) {
            if (handle())
                opencl_rt::clReleaseCommandQueue(handle());
            set(other.handle());
            other.set(nullptr);
        }
        return *this;
    }
    ~command_queue()
    {
        if (handle())
            opencl_rt::clReleaseCommandQueue(handle());
    }
    template <size_t N>
    void enqueue(kernel& k, const std::array<size_t, N>& global_work_size, const std::array<size_t, N>& local_work_size)
    {
        static_assert(N > 0 && N <= 3);
        //TODO: events
        opencl_rt::clEnqueueNDRangeKernel(handle(), k.handle(), N, nullptr, global_work_size.data(), local_work_size.data(), 0, nullptr, nullptr);
    }
    void finish()
    {
        opencl_rt::clFinish(handle());
    }

    void enqueue_blocking_read(buffer& buff, size_t offset_in_buff, size_t size, void* destination)
    {
        auto result = this->enqueue_read(buff, true, offset_in_buff, size, destination);
        if (result != CL_SUCCESS)
            THROW_ERROR(result);
    }

private:
    cl_int enqueue_read(buffer& buff, bool block, size_t offset_in_buff, size_t size, void* destination)
    {
        return opencl_rt::clEnqueueReadBuffer(handle(), buff.handle(), block, offset_in_buff, size, destination, 0, nullptr, nullptr);
    }

    static cl_command_queue create(context& ctx, device& dev)
    {
        //TODO: profiling
        cl_int error_code = 0;
        auto result = opencl_rt::clCreateCommandQueue(ctx.handle(), dev.handle(), 0, &error_code);
        if (error_code != CL_SUCCESS)
            THROW_ERROR(error_code);
        return result;
    }
};

class program : public backend<cl_program> {
    using kernel_class = kernel;

public:
    NON_COPYABLE(program);

    explicit program(context& ctx, const std::string& source)
        : backend(create(ctx, source))
    {
    }
    program(program&& other) noexcept
        : backend(other.handle())
    {
        other.set(nullptr);
    }
    program& operator=(program&& other) noexcept
    {
        if (handle() != other.handle()) {
            if (handle())
                opencl_rt::clReleaseProgram(handle());
            this->set(other.handle());
            other.set(nullptr);
        }
        return *this;
    }
    ~program()
    {
        if (handle())
            opencl_rt::clReleaseProgram(handle());
    }
    void build()
    {
        auto result = opencl_rt::clBuildProgram(handle(), 0, nullptr, nullptr, nullptr, nullptr);
        if (result != CL_SUCCESS)
            THROW_ERROR(result);
    }
    [[nodiscard]] std::string buildLog(device& dev)
    {
        std::string result;
        size_t size = 0;
        opencl_rt::clGetProgramBuildInfo(handle(), dev.handle(), CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
        result.resize(size);
        opencl_rt::clGetProgramBuildInfo(handle(), dev.handle(), CL_PROGRAM_BUILD_LOG, size, &result[0], nullptr);
        return result;
    }
    kernel_class kernel(const std::string& name)
    {
        return kernel_class(opencl_rt::clCreateKernel(handle(), name.c_str(), nullptr));
    }

private:
    static cl_program create(context& ctx, const std::string& source)
    {
        cl_int error_code = 0;
        auto cstr = source.c_str();
        auto result = opencl_rt::clCreateProgramWithSource(ctx.handle(), 1, &cstr, nullptr, &error_code);
        if (error_code != CL_SUCCESS)
            THROW_ERROR(error_code);
        return result;
    }
};

[[nodiscard]] inline std::vector<platform> platforms()
{
    cl_uint numPlatforms = 0;
    opencl_rt::clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::vector<cl_platform_id> platforms;
    platforms.resize(numPlatforms);
    opencl_rt::clGetPlatformIDs(platforms.size(), &platforms[0], nullptr);
    std::vector<platform> result;
    result.reserve(platforms.size());
    for (auto p : platforms)
        result.emplace_back(p);
    return result;
}
}

#undef DECLARE_CL_API
#undef LOAD_CL_API
#undef THROW_ERROR
