#pragma once

#include "so_loader.hpp"

#include <CL/cl.h>
#include <string>
#include <vector>

#define DECLARE_CL_API(name) inline decltype (::name)* name = nullptr
#define LOAD_CL_API(name) name = reinterpret_cast<decltype(name)>(library_handle.symbol(#name))

namespace opencl_rt {

    inline so_loader::library library_handle;
    DECLARE_CL_API(clGetPlatformIDs);
    DECLARE_CL_API(clGetPlatformInfo);
    DECLARE_CL_API(clGetDeviceIDs);
    DECLARE_CL_API(clGetDeviceInfo);
    DECLARE_CL_API(clCreateContext);
    DECLARE_CL_API(clReleaseContext);

    bool load(const std::string& libraryPath)
    {
        library_handle = so_loader::library(libraryPath);
        if (library_handle.is_open()) {
            LOAD_CL_API(clGetPlatformIDs);
            LOAD_CL_API(clGetPlatformInfo);
            LOAD_CL_API(clGetDeviceIDs);
            LOAD_CL_API(clGetDeviceInfo);
            LOAD_CL_API(clCreateContext);
            LOAD_CL_API(clReleaseContext);
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
        explicit backend(T h) noexcept : _handle(h) {}
        const auto& handle() const noexcept { return _handle; }
    protected:
        void set(T value) noexcept { _handle = value; }
    private:
        T _handle;
    };

    template <int param>
    struct device_param_trait;

    template <>
    struct device_param_trait<CL_DEVICE_NAME> { using type = std::string; };
    template <>
    struct device_param_trait<CL_DRIVER_VERSION> { using type = std::string; };
    template <>
    struct device_param_trait<CL_DEVICE_VERSION> { using type = std::string; };
    template <>
    struct device_param_trait<CL_DEVICE_AVAILABLE> { using type = cl_bool; };
    template <>
    struct device_param_trait<CL_DEVICE_PLATFORM> { using type = cl_platform_id; };

    namespace priv {
        template <typename T>
        void resize(T& /*param*/, size_t /*size*/) {}
        template <>
        void resize<std::string>(std::string& param, size_t size)
        {
            param.resize(size);
        }

        template <typename T>
        void* address(T& param) { return &param; }
        template <>
        void* address<std::string>(std::string& param) { return &param[0]; }
    }

    class device : public backend<cl_device_id>{
    public:
        using backend::backend;

        template <int param>
        auto info() const {
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

        std::string info(cl_platform_info param) const {
            size_t size = 0;
            opencl_rt::clGetPlatformInfo(handle(), param, 0, nullptr, &size);
            std::string result(size, ' ');
            opencl_rt::clGetPlatformInfo(handle(), param, result.size(), &result[0], nullptr);
            return result;
        }

        std::vector<device> devices(cl_device_type type) const {
            size_t numDevices = 0;
            opencl_rt::clGetDeviceIDs(handle(), type, 0, nullptr, &numDevices);
            std::vector<cl_device_id> devices(numDevices);
            opencl_rt::clGetDeviceIDs(handle(), type, numDevices, &devices[0], nullptr);
            std::vector<device> result;
            for (auto d : devices)
                result.emplace_back(d);
            return result;
        }
    };

    class context : public backend<cl_context>{
    public:
        using backend::backend;

        explicit context(device& dev) : backend(create(dev)) {

        }
        context(context&& other) noexcept : backend(other.handle()){
            other.set(nullptr);
        }
        context& operator = (context&& other) noexcept {
            if (handle() != other.handle()) {
                if (handle())
                    opencl_rt::clReleaseContext(handle());
                set(other.handle());
                other.set(nullptr);
            }
            return *this;
        }
        ~context() {
            if (handle())
                opencl_rt::clReleaseContext(handle());
        }

        context(const context&) = delete;
        context& operator= (const context&) = delete;

    private:
        static cl_context create(device& dev) {
            cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                                  reinterpret_cast<cl_context_properties>(dev.info<CL_DEVICE_PLATFORM>()),
                                                  0};
            cl_int error_code = 0;
            auto result = opencl_rt::clCreateContext(properties, 1, &dev.handle(), nullptr, nullptr, &error_code);
            if (error_code != CL_SUCCESS)
                throw error_code; // TODO
            return result;
        }
    };

    std::vector<platform> platforms()
    {
        size_t numPlatforms = 0;
        opencl_rt::clGetPlatformIDs(0, nullptr, &numPlatforms);
        std::vector<cl_platform_id> platforms;
        platforms.resize(numPlatforms);
        opencl_rt::clGetPlatformIDs(platforms.size(), &platforms[0], nullptr);
        std::vector<platform> result;
        for (auto p : platforms)
            result.emplace_back(p);
        return result;
    }
}

#undef DECLARE_CL_API
#undef LOAD_CL_API
