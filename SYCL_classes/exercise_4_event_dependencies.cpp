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
	constexpr size_t DATA_SIZE{16};
	std::vector<float> a;
	std::vector<float> b;
	std::vector<float> c;
	std::vector<float> result;

	a.resize(DATA_SIZE);
	b.resize(a.size());
	c.resize(a.size());
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
		auto pC = sycl::experimental::usm_wrapper<float>(sycl::malloc_device<float>(c.size(), q));
		auto pResult = sycl::experimental::usm_wrapper<float>(sycl::malloc_device<float>(result.size(), q));

		auto e0 = q.submit([&](sycl::handler& cgh) {
			cgh.parallel_for<class usm_add>(
				sycl::range<1>{a.size()},
				[=](sycl::id<1> idx) {
					pA[idx[0]] = idx[0];
				});
			});

		auto e1 = q.submit([&](sycl::handler& cgh) {
			cgh.depends_on(e0);
			cgh.parallel_for<class usm_bufferB>(
				sycl::range<1>{b.size()},
				[=](sycl::id<1> idx) {
					pB[idx[0]] = pA[idx[0]] * 2;
				}
			);
		});

		auto e2 = q.submit([&](sycl::handler& cgh) {
			cgh.depends_on(e0);
			cgh.parallel_for<class usm_bufferC>(
				sycl::range<1>{c.size()},
				[=](sycl::id<1> idx) {
					pC[idx[0]] = pA[idx[0]] * 3;
				}
			);
		});

		auto e3 = q.submit([&](sycl::handler& cgh) {
			cgh.depends_on({ e1, e2 });
			cgh.parallel_for<class usm_bufferFinal>(
				sycl::range<1>{b.size()},
				[=](sycl::id<1> idx) {
					pResult[idx[0]] = pB[idx[0]] + pC[idx[0]];
				}
			);
		});

		auto e4 = q.memcpy(result.data(), pResult, result.size() * sizeof(float), e3);

		e4.wait();

		sycl::free(pResult, q);
		sycl::free(pA, q);
		sycl::free(pB, q);
		sycl::free(pC, q);
	}

	std::copy(result.begin(), result.end(), std::ostream_iterator<float>(std::cout, ", "));

	return 0;
}
