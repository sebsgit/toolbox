/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
*/

#include <algorithm>
#include <iostream>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <benchmark.h>
#include <image_conv.h>

#include <SYCL/sycl.hpp>

class image_convolution;

constexpr util::filter_type filterType = util::filter_type::blur;
constexpr int filterWidth = 11;
constexpr int halo = filterWidth / 2;

TEST_CASE("image_convolution_naive", "image_convolution_reference") {
    const char* inputImageFile =
        "C:/Users/seb/sycl_classes/sycl_academy_recursive/syclacademy/Code_Exercises/Images/dogs.png";
    const char* outputImageFile =
        "C:/Users/seb/sycl_classes/sycl_academy_recursive/syclacademy/Code_Exercises/Images/blurred_dogs.png";

    auto inputImage = util::read_image(inputImageFile, halo);

    auto outputImage = util::allocate_image(
        inputImage.width(), inputImage.height(), inputImage.channels());

    auto filter = util::generate_filter(util::filter_type::blur, filterWidth);

    try {
        sycl::queue myQueue{ sycl::cpu_selector(),
                            [](sycl::exception_list exceptionList) {
                              for (auto e : exceptionList) {
                                std::rethrow_exception(e);
                              }
                            } };

        std::cout << "Running on "
            << myQueue.get_device().get_info<sycl::info::device::name>()
            << "\n";

        auto inputImgWidth = inputImage.width();
        auto inputImgHeight = inputImage.height();
        auto channels = inputImage.channels();
        auto filterWidth = filter.width();
        auto halo = filter.half_width();

        auto globalRange = sycl::range<2>(inputImgWidth, inputImgHeight);
        auto localRange = sycl::range<2>(16, 16);
        auto ndRange = sycl::nd_range<2>(globalRange, localRange);

        auto inBufRange = (inputImgWidth + (halo * 2)) * sycl::range<2>(1, channels);
        auto outBufRange = inputImgHeight * sycl::range<2>(1, channels);
        auto filterRange = filterWidth * sycl::range<2>(1, channels);

        {
            auto inBuf = sycl::buffer<float, 2>{ inputImage.data(), inBufRange };
            auto outBuf = sycl::buffer<float, 2>{ outBufRange };
            auto filterBuf = sycl::buffer<float, 2>{ filter.data(), filterRange };
            outBuf.set_final_data(outputImage.data());

            auto inBuffVec4 = inBuf.reinterpret<sycl::float4>(inBufRange / sycl::range<2>(1, 4));
            auto outBuffVec4 = outBuf.reinterpret<sycl::float4>(outBufRange / sycl::range<2>(1, 4));

            util::benchmark(
                [&]() {
                    myQueue.submit([&](sycl::handler& cgh) {
                        auto inputAcc = inBuffVec4.get_access<sycl::access::mode::read>(cgh);
                        auto outputAcc =
                            outBuffVec4.get_access<sycl::access::mode::write>(cgh);
                        auto filterAcc =
                            filterBuf.get_access<sycl::access::mode::read>(cgh);
                        auto localMem = 
                            sycl::accessor<sycl::float4, 2, sycl::access::mode::read_write, sycl::access::target::local>(localRange + halo * 2, cgh);

                        cgh.parallel_for<image_convolution>(
                            ndRange, [=](sycl::nd_item<2> item) {
                                auto globalId = item.get_global_id();
                                globalId = {globalId[1], globalId[0]};

                                auto haloOffset = sycl::id<2>(halo, halo);
                                auto src = (globalId + haloOffset);
                                auto dest = globalId;

                                auto localID = item.get_local_id() + halo;
                                localMem[localID] = inputAcc[src];
                                item.barrier(cl::sycl::access::fence_space::local_space);

                                auto localSrc = localID + haloOffset;

                                sycl::float4 sum = { 0.0f, 0.0f, 0.0f, 0.0f };

                                for (int r = 0; r < filterWidth; ++r) {
                                    for (int c = 0; c < filterWidth; ++c) {
                                        auto srcOffset =
                                            sycl::id<2>(localSrc[0] + (r - halo),
                                                localSrc[1] + ((c - halo)));
                                        auto filterOffset = sycl::id<2>(r, c);

                                        sum += localMem[srcOffset] * filterAcc[filterOffset];
                                    }
                                }

                                outputAcc[dest] = sum;
                            });
                        });

                    myQueue.wait_and_throw();
                },
                100, "image convolution (coalesced)");
        }
    }
    catch (sycl::exception e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    }

    util::write_image(outputImage, outputImageFile);

    REQUIRE(true);
}
