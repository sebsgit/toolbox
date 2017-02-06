#pragma once

#include "ctml/utils.hpp"
#include "ctml/sequence_iterator.hpp"
#include <type_traits>
#include <cstdint>

namespace ctml {

namespace random_distributions {
    template <typename Generator, typename DistributionParameters, uint64_t seed = random_seed<uint64_t>::value, typename SeqInit = typename ctml::sequence_iterator<Generator, seed>::next>
    class uniform {
    public:
        static constexpr auto a = DistributionParameters::a();
        static constexpr auto b = DistributionParameters::b();
    private:
        static constexpr auto value_helper() {
            static_assert(a <= b, "error: cannot generate uniform distribution with b < a");
            using generator = typename SeqInit::generator;
            return normalize<generator>(SeqInit::value) * (b - a) + a;
        }
    public:
        using next = uniform<Generator, DistributionParameters, seed, typename SeqInit::next>;
        static constexpr auto value = value_helper();
    };

    template <typename Generator, long a_, long b_, uint64_t seed = random_seed<uint64_t>::value, typename SeqInit = typename ctml::sequence_iterator<Generator, seed>::next>
    class uniform_discrete {
    public:
        static constexpr auto a = a_;
        static constexpr auto b = b_;
    private:
        static constexpr typename Generator::type value_helper() {
            static_assert(a <= b, "error: cannot generate uniform distribution with b < a");
            using generator = typename SeqInit::generator;
            return normalize<generator>(SeqInit::value) * (b - a) + a;
        }
    public:
        using next = uniform_discrete<Generator, a, b, seed, typename SeqInit::next>;
        static constexpr auto value = value_helper();
    };

    template <typename Generator, typename Params, uint64_t seed = random_seed<uint64_t>::value, typename SeqInit = typename ctml::sequence_iterator<Generator, seed>::next>
    class bernoulli {
    private:
        static constexpr bool value_helper() {
            using generator = typename SeqInit::generator;
            constexpr double p = Params::p();
            static_assert(p <= 1.0 && p >= 0, "error: probability parameter in bernoulli distribution needs to be in range [0,1]");
            return normalize<generator>(SeqInit::value) < p;
        }
    public:
        using next = bernoulli<Generator, Params, seed, typename SeqInit::next>;
        static constexpr bool value = value_helper();
    };
}
}
