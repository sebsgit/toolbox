////////////////////////////////////////////////////////////////////////////////
//
//  SYCL Sample Code
//
//  Author: Codeplay, April 2019
//
////////////////////////////////////////////////////////////////////////////////

/*!
If you encounter an application cannot start because SYCL.dll or SYCL_d.dll is
missing error, then you may have to restart your computer in order to update the
environment variable COMPUTECPPROOT.
*/

#include <SYCL/sycl.hpp>

#include <iostream>
#include <vector>

using namespace sycl;

int main(int argc, char *argv[]) {
	std::vector<float> input{10, 11, 12, 13};
	std::vector<float> target;

	target.resize(input.size());

	{
		queue q{ cpu_selector() };

		auto buffIn = buffer<float>(input.data(), range<1>{input.size()});
		buffIn.set_final_data(nullptr);

		auto buffTemp = buffer<float>(range<1>{input.size()});
		buffTemp.set_final_data(nullptr);

		auto buffTarget = buffer<float>(range<1>{input.size()});
		buffTarget.set_final_data(target.data());

		auto ev1 = q.submit([&](handler &cgh) {
			auto accIn = buffIn.get_access<access::mode::read>(cgh);
			auto accTemp = buffTemp.get_access<access::mode::discard_write>(cgh);

			cgh.parallel_for<class kenrA>(
				range<1>(input.size()),
				[=](id<1> idx) { accTemp[idx] = accIn[idx] * 2; }
			);
		});

		auto ev2 = q.submit([&](handler& cgh) {
			cgh.depends_on(ev1);
			auto accTemp = buffTemp.get_access<access::mode::read>(cgh);
			auto accFinal = buffTarget.get_access<access::mode::discard_write>(cgh);

			cgh.parallel_for<class kenrB>(
				range<1>(input.size()),
				[=](id<1> idx) { accFinal[idx] = accTemp[idx] + 1; }
			);
		});

		event::wait_and_throw({ev1, ev2});
	}

	std::copy(target.begin(), target.end(), std::ostream_iterator<float>(std::cout, ", "));

	return 0;
}

////////////////////////////////////////////////////////////////////////////////