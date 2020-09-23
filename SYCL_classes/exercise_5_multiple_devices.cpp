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

using namespace sycl;

int main(int argc, char *argv[]) {
	try {
		auto device_0{ cpu_selector().select_device() };
		auto device_1{ host_selector().select_device() };

		std::vector<float> init_data{0, 1, 2, 3};
		std::vector<float> target_data;
		target_data.resize(init_data.size());
		{

			queue q0{ device_0 };
			queue q1{ device_1 };

			auto b_init = buffer<float>{ init_data.data(), range<1>{init_data.size()} };
			b_init.set_write_back(false);
			auto b_result = buffer<float>{ range<1> {target_data.size()} };
			b_result.set_final_data(target_data.data());
			auto b_temp = buffer<float>{ range<1> {target_data.size()} };
			b_temp.set_write_back(false);

			auto e1 = q0.submit([&](handler& cgh) {
				auto acc_init = b_init.get_access<access::mode::read>(cgh);
				auto acc_tmp = b_temp.get_access<access::mode::discard_write>(cgh);

				cgh.parallel_for<class kenr1>(
					range<1> {init_data.size()},
					[=](id<1> idx) {
						acc_tmp[idx] = 2 * acc_init[idx];
					}
				);
			});

			auto e2 = q1.submit([&](handler &cgh) {
				cgh.depends_on(e1);
				auto acc_tmp = b_temp.get_access<access::mode::read>(cgh);
				auto acc_res = b_result.get_access<access::mode::discard_write>(cgh);

				cgh.parallel_for<class kenr21>(
					range<1> {init_data.size()},
					[=](id<1> idx) {
						acc_res[idx] = 2 * acc_tmp[idx];
					}
				);
			});

			e2.wait_and_throw();
		}
		std::copy(target_data.begin(), target_data.end(), std::ostream_iterator<float>(std::cout, ", "));
	}
	catch (sycl::exception exc) {
		std::cout << "SYCL EXC: " << exc.what();
	}
	return 0;
}

////////////////////////////////////////////////////////////////////////////////