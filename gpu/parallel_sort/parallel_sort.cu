//
// Sorting random float vector on GPU using a distributed mergesort.
//
// The algorithm is divided in the following steps:
//   0) [CPU] generate the partitioning data on the host (static per input size),
//            describes which thread should work at which offset of the input
//   1) [GPU] use a sorting network to sort chunks of 8 elements of the input data
//   2) [GPU] continuously merge the sorted two data chunks into one : [8, 8] -> [16] >> [16,16] -> [32] >> ..., 
//            until the full input data is sorted, step 2) uses the static partitioning data from step 0)
//
// references: 
//  - "Merge Path - A Visually Intuitive Approach to Parallel Merging" (Oded Green, Saher Odeh, Yitzhak Birk)
//  - "Sorting Networks", http://staff.ustc.edu.cn/~csli/graduate/algorithms/book6/chap28.htm
//

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <iostream>
#include <ctime>
#include <vector>
#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "parallel_sort.h"

template <typename T>
thrust::host_vector<T> randomVector(size_t size)
{
    thrust::host_vector<T> result;
    result.reserve(size);
    for (size_t i = 0; i < size; ++i)
    {
        result.push_back(static_cast<T>(rand() % 50 - 25));
    }
    return result;
}

struct fill_part_params_thin
{
    uint16_t off_a;
};

// compare-and-swap
template <typename T> __device__
void cas(T &x, T &y) noexcept
{
    if (x > y)
    {
        const T t{x};
        x = y;
        y = t;
    }
}

///
/// @brief sorts chunks of length 8 via sorting network
///
template <typename T> __global__
void presort_8x(const T* a, const size_t len_a, const size_t off_a, T* res, cudaTextureObject_t a_as_tex)
{
    const auto flat_thread_id{ blockIdx.x * blockDim.x + threadIdx.x };
    const size_t offset{ off_a + flat_thread_id * 8 };
    if (offset < len_a)
    {
        auto x1{ tex1Dfetch<T>(a_as_tex, offset + 0) };
        auto x2{ tex1Dfetch<T>(a_as_tex, offset + 1) };
        auto x3{ tex1Dfetch<T>(a_as_tex, offset + 2) };
        auto x4{ tex1Dfetch<T>(a_as_tex, offset + 3) };
        auto x5{ tex1Dfetch<T>(a_as_tex, offset + 4) };
        auto x6{ tex1Dfetch<T>(a_as_tex, offset + 5) };
        auto x7{ tex1Dfetch<T>(a_as_tex, offset + 6) };
        auto x8{ tex1Dfetch<T>(a_as_tex, offset + 7) };
        
        cas(x1, x2);
        
        cas(x2, x3);

        cas(x1, x2);
        cas(x3, x4);

        cas(x2, x3);
        cas(x4, x5);

        cas(x1, x2);
        cas(x3, x4);
        cas(x5, x6);

        cas(x2, x3);
        cas(x4, x5);
        cas(x6, x7);

        cas(x1, x2);
        cas(x3, x4);
        cas(x5, x6);
        cas(x7, x8);

        cas(x2, x3);
        cas(x4, x5);
        cas(x6, x7);

        cas(x1, x2);
        cas(x3, x4);
        cas(x5, x6);

        cas(x2, x3);
        cas(x4, x5);

        cas(x1, x2);
        cas(x3, x4);

        cas(x2, x3);

        cas(x1, x2);

        res[offset + 0] = x1;
        res[offset + 1] = x2;
        res[offset + 2] = x3;
        res[offset + 3] = x4;
        res[offset + 4] = x5;
        res[offset + 5] = x6;
        res[offset + 6] = x7;
        res[offset + 7] = x8;
    }
}

template <typename T> __global__
void apply_all_multi_element(const T* a, const size_t len_a, T* res, size_t pow2_1, const size_t elements_per_thread)
{
    const auto flat_thread_id{ blockIdx.x * blockDim.x + threadIdx.x };
    const size_t n_proc_per_this_thread{ gridDim.x * blockDim.x / pow2_1 };
    
    const size_t effective_pid{
        (n_proc_per_this_thread > 1 ? flat_thread_id % n_proc_per_this_thread : 0)
    };
    const auto idx =
        parallel_sort::merge_path_intersection(a, len_a, a, len_a, effective_pid, elements_per_thread);
    parallel_sort::fill_from(idx, a, len_a, a, len_a, elements_per_thread, res, flat_thread_id * elements_per_thread);
}


template <typename T> __global__
void apply_all_multi_job(const fill_part_params_thin* parts, const size_t parts_count,
    const T* a, const size_t len_a, T* res, 
    const size_t jobs_per_thread, const size_t elements_per_thread, const size_t total_elements_per_thread)
{
    const auto flat_thread_id{ blockIdx.x * blockDim.x + threadIdx.x };
    const fill_part_params_thin* local_parts = parts + flat_thread_id * jobs_per_thread;
    for (size_t i = 0; i < jobs_per_thread; ++i)
    {
        const fill_part_params_thin& p{ local_parts[i] };
        const size_t effective_offset{
            flat_thread_id * total_elements_per_thread + i * elements_per_thread
        };
        const auto a_off{p.off_a};
        const auto b_off{a_off + elements_per_thread / 2};
        parallel_sort::fill_from({0, 0}, a + a_off, len_a, a + b_off, len_a, elements_per_thread, res, effective_offset);
    }
}


#define CHECK_ERR(status) if (status != cudaSuccess) \
    { \
        std::cout << "error: " << cudaGetErrorString(status) << '\n'; \
        return -1; \
    }

int main()
{
    srand(time(nullptr));

    auto status{cudaSetDevice(0)};
    int dev_id{ 0 };
    cudaGetDevice(&dev_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, dev_id);

    int managed_mem{0};
    status = cudaDeviceGetAttribute(&managed_mem, cudaDevAttrManagedMemory, dev_id);
    CHECK_ERR(status);

    std::cout << "Device: " << props.name << '\n';
    std::cout << "managed ?: " << managed_mem << '\n';
    std::cout << "constant memory: " << (props.totalConstMem / 1024) << " Kb\n";
    std::cout << "max texture size 1D: " << (props.maxTexture1D) << '\n';
    
    //
    // RTX 2080 results:
    // 1024 * 1024 || 512 x 1 --> 120ms / 41120 Kb [base version]
    // 1024 * 1024 || 32p x 16b --> 22ms / 2112 Kb [optimizations: sorting network x8, block / grid size adjustments]
    // 1024 * 1024 || 32p x 16b --> 17ms / 2112 Kb [optimizations: texture 1D for read-only data]
    // 1024 * 1024 || 16p x 64b --> 12ms / 2112 Kb
    // 1024 * 1024 || 8 x 128 --> 11ms / 2112 Kb
    // 1024 * 1024 || 8 x 128 --> 11ms / 274 Kb
    // 1024 * 1024 || 8 x 128 --> 11ms / 252 Kb
    //

    const size_t data_size{1024 * 1024};
    const size_t num_processors{ 8 };
    const size_t num_blocks{ 128 };
    const size_t effective_thread_count{num_processors * num_blocks};
    const auto num_threads_for_presort{ props.maxThreadsPerBlock };
    const size_t num_blocks_for_presort{32};
    const size_t presorted_chunks_size{8};

    {
        int min_grid_s{ 0 };
        int min_block_s{0};
        status = cudaOccupancyMaxPotentialBlockSize(&min_grid_s, &min_block_s, presort_8x<float>, 0, data_size);
        CHECK_ERR(status);
        std::cout << "Presort: selected " << num_blocks_for_presort << " x " << num_threads_for_presort << '\n';
        std::cout << "\tsuggested: " << min_grid_s << " x " << min_block_s << '\n';

        status = cudaOccupancyMaxPotentialBlockSize(&min_grid_s, &min_block_s, apply_all_multi_element<float>, 0, data_size);
        CHECK_ERR(status);
        std::cout << "Muti-element pass: selected " << num_blocks << " x " << num_processors << '\n';
        std::cout << "\tsuggested: " << min_grid_s << " x " << min_block_s << '\n';

        status = cudaOccupancyMaxPotentialBlockSize(&min_grid_s, &min_block_s, apply_all_multi_job<float>, 0, data_size);
        CHECK_ERR(status);
        std::cout << "Multi-job pass: selected " << num_blocks << " x " << num_processors << '\n';
        std::cout << "\tsuggested: " << min_grid_s << " x " << min_block_s << '\n';

        const auto bookkeping_mem_single_kernel_call{ num_blocks * num_processors * sizeof(fill_part_params_thin) };
        std::cout << "mem needed for bookeeping, single kernel call: " << bookkeping_mem_single_kernel_call / 1024 << " Kb\n";
    }

    auto data_to_sort{randomVector<float>(data_size)};

    {
        thrust::device_vector<float> dev_input = data_to_sort;
        thrust::device_vector<float> dev_output;
        dev_output.resize(data_size);

        thrust::host_vector<float> host_result;
        host_result.resize(data_size);

        cudaResourceDesc input_res_desc{};
        input_res_desc.resType = cudaResourceTypeLinear;
        input_res_desc.res.linear.devPtr = thrust::raw_pointer_cast(dev_input.data());
        input_res_desc.res.linear.sizeInBytes = data_size * sizeof(float);
        input_res_desc.res.linear.desc.f = cudaChannelFormatKindFloat;
        input_res_desc.res.linear.desc.x = 32;

        cudaTextureDesc input_tex_desc{};
        input_tex_desc.filterMode = cudaFilterModePoint;
        input_tex_desc.readMode = cudaReadModeElementType;

        cudaTextureObject_t input_as_tex{};
        status = cudaCreateTextureObject(&input_as_tex, &input_res_desc, &input_tex_desc, nullptr);
        CHECK_ERR(status);

        const size_t total_number_of_steps{static_cast<size_t>(std::log2(data_size))};
        size_t total_bookkeeping_memory{0};

        struct offset_data_t
        {
            thrust::device_vector<fill_part_params_thin> data;
        };

        std::vector<offset_data_t> offset_data_dev;
        {
            const auto work_partitioning_start_ts{ std::chrono::high_resolution_clock::now() };
            const auto work_partitions{ 
                parallel_sort::create_sort_partitions(data_size, effective_thread_count, presorted_chunks_size) 
            };
            const auto work_partitioning_end_ts{ std::chrono::high_resolution_clock::now() };

            std::cout << "partitioning the work took: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(work_partitioning_end_ts - work_partitioning_start_ts).count()
                << " ms.\n";

            for (auto &work_slice : work_partitions)
            {
                const auto& data{ work_slice.data() };
                thrust::host_vector<fill_part_params_thin> data_for_iterations;
                for (auto& d : data)
                {
                    if (d.parts.size() > 1)
                    {
                        for (auto x : d.parts) {
                            fill_part_params_thin x_0;
                            x_0.off_a = x.off_a;
                            data_for_iterations.push_back(x_0);
                            total_bookkeeping_memory += sizeof(x_0);
                        }
                    }
                }
                if (!data_for_iterations.empty())
                {
                    offset_data_dev.push_back(offset_data_t{});
                    offset_data_dev.back().data = std::move(data_for_iterations);
                }
            }
        }

        std::cout << "Mem for bookkeeping: " << (total_bookkeeping_memory / (1024)) << " Kb\n";
        std::cout << "Mem for data: " << (2 * data_size) / (1024) << "Kb\n";

        const auto gpu_start_ts{ std::chrono::high_resolution_clock::now() };
        auto gpu_stop_ts = gpu_start_ts;

        {
            size_t idx = 0;
            size_t step{ 1 };

            auto* ptr_in{ thrust::raw_pointer_cast(dev_input.data()) };
            auto* ptr_out{ thrust::raw_pointer_cast(dev_output.data()) };

            {
                // pre-sorting chunks of 8 elements
                const size_t num_procs_for_8x{ static_cast<size_t>(num_threads_for_presort) };
                const size_t elems_in_one_call{ num_procs_for_8x * presorted_chunks_size * num_blocks };
                const size_t n_kernel_cals{ elems_in_one_call >= data_size ? 1 : data_size / elems_in_one_call };

                for (size_t i = 0; i < n_kernel_cals; ++i)
                {
                    const size_t data_offset{i * elems_in_one_call};
                    presort_8x <<<num_blocks_for_presort, num_procs_for_8x >>> (ptr_in, data_size, data_offset, ptr_out, input_as_tex);
                }

                cudaMemcpy(ptr_in, ptr_out, data_size * sizeof(float), cudaMemcpyDeviceToDevice);

                idx = static_cast<size_t>(std::log2(presorted_chunks_size));
                step = presorted_chunks_size;
            }

            size_t jobs_per_thread = data_size / (step * 2 * effective_thread_count);
            if (jobs_per_thread == 0)
            {
                jobs_per_thread = 1;
            }

            for (size_t i = idx; i < total_number_of_steps; ++i)
            {
                if (jobs_per_thread > 1)
                {
                    const auto dev_ptr_for_slice_data{ thrust::raw_pointer_cast(offset_data_dev[i - 3].data.data()) };
                    apply_all_multi_job <<< num_blocks, num_processors >>> (
                        dev_ptr_for_slice_data, offset_data_dev[idx].data.size(),
                        ptr_in, step, ptr_out,
                        jobs_per_thread, step * 2, data_size / effective_thread_count);
                }
                else
                {
                    apply_all_multi_element <<< num_blocks, num_processors >>> (
                        ptr_in, step, ptr_out, pow(2, total_number_of_steps - idx - 1),
                        data_size / effective_thread_count);
                }

                step *= 2;
                ++idx;
                if (jobs_per_thread > 1)
                {
                    jobs_per_thread /= 2;
                }

                if (step < data_size)
                    cudaMemcpy(ptr_in, ptr_out, data_size * sizeof(float), cudaMemcpyDeviceToDevice);
            }

            cudaDeviceSynchronize();
            gpu_stop_ts = std::chrono::high_resolution_clock::now();

            status = cudaMemcpy(host_result.data(), ptr_out, data_size * sizeof(float), cudaMemcpyDeviceToHost);

            auto memcpy_stop{std::chrono::high_resolution_clock::now()};

            std::cout << "GPU data transfer (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(memcpy_stop - gpu_stop_ts).count() << '\n';

            CHECK_ERR(status);
        }

        std::cout << "GPU time (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(gpu_stop_ts - gpu_start_ts).count() << '\n';

        if (std::is_sorted(host_result.begin(), host_result.end()))
        {
            std::cout << "sorting length: " << data_size << ", number of threads: " << effective_thread_count << ", success\n";
        }
        else
        {
            std::cout << "sorting failed\n";
            std::copy(host_result.begin(), host_result.end(), std::ostream_iterator<float>(std::cout, ", "));
            std::cout << "\n";
        }

        cudaDestroyTextureObject(input_as_tex);
    } // gpu resources scope end

    const auto cpu_start_ts{ std::chrono::high_resolution_clock::now() };
    auto cpu_data = data_to_sort;
    std::sort(cpu_data.begin(), cpu_data.end());
    const auto cpu_stop_ts{ std::chrono::high_resolution_clock::now() };
    std::cout << "CPU time (ms): "
        << std::chrono::duration_cast<std::chrono::milliseconds>(cpu_stop_ts - cpu_start_ts).count() << '\n';
    
    if (!std::is_sorted(cpu_data.begin(), cpu_data.end()))
    {
        std::cout << "fail on cpu\n";
    }

    status = cudaDeviceReset();
    CHECK_ERR(status);

    return 0;
}
