#ifndef PARALLEL_SORT_H_
#define PARALLEL_SORT_H_

#include <algorithm>
#include <vector>
#include <cassert>
#include <cmath>

#ifndef __NVCC__
#define __device__
#define __host__
#endif

// utilities for parallel sorting
namespace parallel_sort
{
struct indices
{
    size_t a{0};
    size_t b{0};
};

///
/// @brief Implements the "merge path" search algorithm descibed in "Merge Path - A Visually Intuitive Approach to Parallel Merging"
///
template <typename T>
__device__ __host__
indices merge_path_intersection(
    const T * a, const size_t len_a,
    const T * b, const size_t len_b,
    const size_t diag_id,
    const size_t step)
{
    assert(len_b >= len_a);

    if (diag_id == 0)
    {
        return indices{0, 0};
    }

    const size_t diag_start_offset{step * diag_id};
    indices diag_start{0, 0};
    if (diag_start_offset <= len_b)
    {
        diag_start = {0, diag_start_offset};
    }
    else
    {
        diag_start = {diag_start_offset - len_b, len_b};
    }

    const auto point_count{len_a - diag_start.a < diag_start.b ? len_a - diag_start.a : diag_start.b};

    auto ith_point = [&](size_t i)
    {
        return indices{diag_start.a + i, diag_start.b - i};
    };

    size_t start{1};
    size_t end{point_count - 1};

    //
    // finds the first element on the diagonal where there is a change in the merge path:
    //      ...0
    //      ..0.
    //      .1..  <--
    //      1...
    //
    while(start <= end)
    {
        const auto mid{(start + end) / 2};
        const auto pt{ith_point(mid)};
        const auto v_up{a[pt.a - 1] < b[pt.b]};
        const auto v_down{a[pt.a] < b[pt.b - 1]};
        if (v_up && !v_down)
        {
            return pt;
        }
        else if (!v_up)
        {
            --end;
        }
        else
        {
            ++start;
        }
    }

    const auto last_p{ith_point(point_count)};
    if (a[last_p.a - 1] < b[last_p.b])
    {
        return last_p;
    }
    else
    {
        return diag_start;
    }
}

template <typename T>
__device__ __host__
void fill_from(
    indices idx,
    const T * a, const size_t len_a,
    const T * b, const size_t len_b,
    const size_t part_size,
    T * res, const size_t r_off)
{
    for(size_t cnt = 0; cnt < part_size; ++cnt)
    {
        if (idx.a == len_a)
        {
            res[r_off + cnt] = b[idx.b++];
        }
        else if (idx.b == len_b)
        {
            res[r_off + cnt] = a[idx.a++];
        }
        else
        {
            if (a[idx.a] < b[idx.b])
            {
                res[r_off + cnt] = a[idx.a];
                idx.a++;
            }
            else
            {
                res[r_off + cnt] = b[idx.b];
                idx.b++;
            }
        }
    }
}

struct fill_part_params
{
    size_t pid;
    size_t off_a;
};

struct parts_per_processor
{
    size_t pid{0};
    std::vector<fill_part_params> parts;
};

// keeps the offset data used in distributed merge for each processor
class SortPartition
{
public:
    void add(const fill_part_params &p)
    {
        auto it{std::find_if(parts_.begin(), parts_.end(), [&](auto &from_vec) {
            return start_pid_ + p.pid == from_vec.pid;
        })};
        if (it == parts_.end())
        {
            parts_.push_back({start_pid_ + p.pid, {}});
            parts_.back().parts.push_back(p);
        }
        else
        {
            it->parts.push_back(p);
        }
    }

    const auto &data() const noexcept
    {
        return parts_;
    }

    void setStartPID(const size_t pid) noexcept
    {
        start_pid_ = pid;
    }

private:
    std::vector<parts_per_processor> parts_;
    size_t start_pid_{0};
};

void merge2(const size_t len_a, const size_t off_a,
            const size_t len_b, const size_t off_b,
            const size_t off_res,
            const size_t proc_count,
            SortPartition &work_partition)
{
    if (len_a == 1 && len_b == 1)
    {
        work_partition.add(fill_part_params{0, off_a});
        return;
    }

    assert(len_b % 2 == 0);
    assert(len_a % 2 == 0);
    assert((len_a + len_b) % proc_count == 0);
    assert(len_a <= len_b);

    for (size_t p = 0; p < proc_count; ++p)
    {
        work_partition.add(fill_part_params{p, off_a});
    }
}

[[nodiscard]] SortPartition merge_pairs(const size_t len, const size_t element_count, size_t proc_count)
{
    assert(len % element_count == 0);
    const size_t num_parts{len / (2 * element_count)};
    const auto step = element_count;

    SortPartition work_partition;

    if (num_parts > proc_count)
    {
        assert(num_parts % proc_count == 0);
        const size_t parts_per_proc{num_parts / proc_count};
        for (size_t p = 0; p < proc_count; ++p)
        {
            const auto proc_off{p * parts_per_proc * step * 2};
            work_partition.setStartPID(p);
            for (size_t n = 0; n < parts_per_proc; ++n)
            {
                const auto batch_start{n * step * 2 + proc_off};
                merge2(element_count, batch_start,
                       element_count, batch_start + element_count,
                       batch_start,
                       1,
                       work_partition);
            }
        }
    }
    else
    {
        assert(proc_count % num_parts == 0);
        const size_t procs_per_part{proc_count / num_parts};
        for (size_t n = 0; n < num_parts; ++n)
        {
            const auto part_start{n * step * 2};
            work_partition.setStartPID(n * procs_per_part);
            merge2(step, part_start, step, part_start + step, n * step * 2, procs_per_part, work_partition);
        }
    }

    return work_partition;
}

bool isPartitionEven(const SortPartition &partition)
{
    const auto &data{partition.data()};
    const size_t proc_count {data.size()};
    if (proc_count == 0)
    {
        return true;
    }

    const size_t workload_proc_0{data[0].parts.size()};
    for (size_t j = 1; j < proc_count; ++j)
    {
        if (data[j].parts.size() != workload_proc_0)
        {
            return false;
        }
    }

    return true;
}

[[nodiscard]]
std::vector<SortPartition> create_sort_partitions(const size_t len, const size_t proc_count, size_t start_step = 1)
{
    std::vector<SortPartition> result;
    result.reserve(static_cast<size_t>(std::log2(len)));
    size_t step{start_step};
    while (step < len)
    {
        auto work_slices = merge_pairs(len, step, proc_count);
        result.push_back(std::move(work_slices));
        step *= 2;
    }

    return result;
}
} // namespace parallel_sort

#endif // PARALLEL_SORT_H_
