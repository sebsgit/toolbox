#include <CL/sycl.hpp>
#include <iostream>

#include <vector>

class test_p_for;

struct usm_selector : public sycl::device_selector
{
	int operator () (const sycl::device& dev) const override
	{
		if (dev.get_info<sycl::info::device::usm_device_allocations>())
		{
			return 10;
		}
		return -1;
	}
};

int main()
{
	std::vector<float> a{ 0, 1, 2, 3 };
	std::vector<float> b{ -12, 99, 32, 821 };
	std::vector<float> result;
	result.resize(a.size());

	{
		usm_selector cpu_sel;

		std::cout << "device: " << cpu_sel.select_device().get_info<sycl::info::device::name>() << '\n';
		if (!cpu_sel.select_device().get_info<sycl::info::device::usm_device_allocations>())
		{
			std::cout << "No USM support!\n";
			return 0;
		}

		sycl::queue q{ cpu_sel };

		auto pA = sycl::experimental::usm_wrapper<float>(sycl::malloc_device<float>(a.size(), q));
		auto pB = sycl::experimental::usm_wrapper<float>(sycl::malloc_device<float>(b.size(), q));
		auto pResult = sycl::experimental::usm_wrapper<float>(sycl::malloc_device<float>(result.size(), q));

		q.memcpy(pA, a.data(), sizeof(float) * a.size()).wait();
		q.memcpy(pB, b.data(), sizeof(float) * b.size()).wait();

		q.submit([&](sycl::handler &cgh) {
			cgh.parallel_for<class usm_add>(
				sycl::range<1>{a.size()},
				[=](sycl::id<1> idx) {
					pResult[idx[0]] = pA[idx[0]] + pB[idx[0]];
				});
		}).wait_and_throw();

		sycl::free(pA, q);
		sycl::free(pB, q);

		q.memcpy(result.data(), pResult, result.size() * sizeof(float)).wait();
		sycl::free(pResult, q);
	}

	std::copy(result.begin(), result.end(), std::ostream_iterator<float>(std::cout, ", "));

	return 0;
}
