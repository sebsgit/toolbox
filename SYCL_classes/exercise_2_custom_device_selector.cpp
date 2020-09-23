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
#include <CL/sycl.hpp>

#include <iostream>

using namespace cl::sycl;

class scalar_add;
class scalar_add2;

struct custom_selector : public sycl::device_selector
{
    int32_t operator ()(const sycl::device& dev) const override
    {
        if (dev.is_cpu())
        {
            return 10;
        }
        return -1;
    }
};

int main(int argc, char *argv[]) {
    try {
        custom_selector cpu_sel;
        queue q{ cpu_sel, [](exception_list el) { for (auto e : el) std::rethrow_exception(e); } };

        float a{ 3 };
        float b{ 5 };
        float r{ 12332 };

        {
            auto bufA = sycl::buffer<float>{ &a, sycl::range<1>{1} };
            auto bufB = sycl::buffer<float>{ &b, sycl::range<1>{1} };
            auto bufR = sycl::buffer<float>{ &r, sycl::range<1>{1} };

            q.submit([&](sycl::handler& cgh) {
                auto accA = bufA.get_access<sycl::access::mode::read>(cgh);
                auto accB = bufB.get_access<sycl::access::mode::read>(cgh);
                auto accR = bufR.get_access<sycl::access::mode::write>(cgh);

                cgh.single_task<scalar_add>([=]() {
                    accR[0] = accA[0] + accB[0]; });
                }

            ).wait_and_throw();

            q.submit([&](sycl::handler& chg2) {
                auto accA = bufA.get_access<sycl::access::mode::read>(chg2);
                auto accB = bufB.get_access<sycl::access::mode::read>(chg2);
                auto accR = bufR.get_access<sycl::access::mode::write>(chg2);
                chg2.single_task<scalar_add2>([=]() {
                    accR[0] = accR[0] + accB[0]; });
                });

            auto accHost = bufR.get_access<sycl::access::mode::read>();
            std::cout << "host: " << accHost[0] << '\n';
        }

        std::cout << r << '\n';
    }
    catch (std::exception e)
    {
        std::cout << "Exc: " << e.what();
    }
}

////////////////////////////////////////////////////////////////////////////////