#include "ctml/utils.hpp"
#include "ctml/sequence_iterator.hpp"
#include "ctml/random/distributions.hpp"
#include "ctml/random/generators.hpp"

#include <cmath>
#include <iostream>

static void testSequenceIterator() {
    class plus_one {
    public:
        constexpr int operator() (int v) const {
            return v + 1;
        }
    };
    using zero = ctml::sequence_iterator<plus_one>;
    using one = zero::next;
    static_assert(std::is_same<int, one::type>::value, "");
    static_assert(one::value == 1, "");
    static_assert(one::next::value == 2, "");
    static_assert(one::next::next::value == 3, "");
    static_assert(one::next::next::next::value == 4, "");
    static_assert(ctml::advance<ctml::sequence_iterator<plus_one>, 100>::value == 100, "");
    static_assert(ctml::accumulate<ctml::sequence_iterator<plus_one>, 2>::value == 3, "0 + 1 + 2");
    static_assert(ctml::accumulate<ctml::sequence_iterator<plus_one>, 10>::value == 55, "0 + 1 + 2 + ... + 10");

    class times_two {
    public:
        constexpr unsigned operator() (int v) const {
            return v * 2;
        }
    };
    using one_1 = ctml::sequence_iterator<times_two, 1>;
    static_assert(std::is_same<unsigned, one_1::type>::value, "");
    static_assert(one_1::value == 1, "");
    static_assert(one_1::next::value == 2, "");
    static_assert(one_1::next::next::value == 4, "");
    static_assert(one_1::next::next::next::value == 8, "");
    static_assert(ctml::advance<one_1, 0>::value == 1, "");
    static_assert(ctml::advance<one_1, 10>::value == 1024, "");

    class div_by_10 {
    public:
        constexpr short operator() (short v) const {
            return v / 10.0;
        }
    };
    using thousand = ctml::sequence_iterator<div_by_10, 1000>;
    static_assert(std::is_same<short, thousand::type>::value, "");
    static_assert(thousand::value == 1000, "");
    static_assert(thousand::next::value == 100, "");
    static_assert(thousand::next::next::value == 10, "");
    static_assert(thousand::next::next::next::value == 1, "");
}

int main(int argc, char *argv[]) {
    static_assert(ctml::pow<2>(2) == 4, "2 ^ 2");
    static_assert(ctml::pow<3>(2.2) == 2.2 * 2.2 * 2.2, "2.2 ^ 3");
    static_assert(ctml::pow<0>(5.2) == 1, "5.2 ^ 0");
    static_assert(ctml::pow<-2>(2.2) == 1.0 / (2.2 * 2.2), "2.2 ^ -2");
    static_assert(std::pow(2, 2) == 4, "2 ^ -2");
    static_assert(std::pow(2.2, -1) == 1.0 / (2.2), "");
    static_assert(ctml::random_generators::blum_blum_shub<int>()(2) > 0, "bbs");
    static_assert(ctml::random_generators::linear_congruential<int, 2, 3, 5>()(2) == 2, "LCG");
    testSequenceIterator();

    std::cout << "bbs:\n";
    using bbs_seq = ctml::sequence_iterator<ctml::random_generators::blum_blum_shub<unsigned long long>, 23843872834>;
    std::cout << bbs_seq::value << '\n';
    std::cout << ctml::advance<bbs_seq, 1>::value << '\n';
    std::cout << ctml::advance<bbs_seq, 2>::value << '\n';
    std::cout << ctml::advance<bbs_seq, 3>::value << '\n';
    std::cout << ctml::advance<bbs_seq, 4>::value << '\n';
    std::cout << ctml::advance<bbs_seq, 5>::value << '\n';
    std::cout << ctml::advance<bbs_seq, 6>::value << '\n';
    std::cout << ctml::advance<bbs_seq, 7>::value << '\n';

    struct uniform_params {
        static constexpr double a() { return 0.0; }
        static constexpr double b() { return 1.0; }
    };
    using unigen = ctml::random_distributions::uniform<ctml::random_generators::blum_blum_shub<unsigned long long>, uniform_params>;
    std::cout << "uniform: " << unigen::a << ' ' << unigen::b << '\n';
    std::cout << ctml::advance<unigen, 10>::value << '\n';
    std::cout << ctml::advance<unigen, 20>::value << '\n';
    std::cout << ctml::advance<unigen, 30>::value << '\n';
    std::cout << ctml::advance<unigen, 40>::value << '\n';
    std::cout << ctml::advance<unigen, 50>::value << '\n';
    std::cout << ctml::advance<unigen, 60>::value << '\n';
    std::cout << ctml::advance<unigen, 70>::value << '\n';

    constexpr int steps = 350;
    std::cout << "mean: " << ctml::accumulate<unigen, steps>::value / (steps * 1.0f) << ", ideal: " << (unigen::a + unigen::b) * 0.5f <<  '\n';

    using unigen_disc = ctml::random_distributions::uniform_discrete<ctml::random_generators::blum_blum_shub<unsigned long long>, 1, 1000>;
    std::cout << "uniform: " << unigen_disc::a << ' ' << unigen_disc::b << '\n';
    std::cout << ctml::advance<unigen_disc, 10>::value << '\n';
    std::cout << ctml::advance<unigen_disc, 20>::value << '\n';
    std::cout << ctml::advance<unigen_disc, 30>::value << '\n';
    std::cout << ctml::advance<unigen_disc, 40>::value << '\n';
    std::cout << ctml::advance<unigen_disc, 50>::value << '\n';
    std::cout << ctml::advance<unigen_disc, 60>::value << '\n';
    std::cout << ctml::advance<unigen_disc, 70>::value << '\n';
    std::cout << "mean: " << ctml::accumulate<unigen_disc, steps>::value / (steps * 1.0f) << ", ideal: " << (unigen_disc::a + unigen_disc::b) * 0.5f <<  '\n';

    std::cout << "bernoulli:\n";
    struct bernoulli_params { static constexpr double p() { return 0.32; } };
    using bernoulli = ctml::random_distributions::bernoulli<ctml::random_generators::linear_congruential<unsigned long long, 13, 92, 43>, bernoulli_params>;
    std::cout << ctml::advance<bernoulli, 10>::value << '\n';
    std::cout << ctml::advance<bernoulli, 20>::value << '\n';
    std::cout << ctml::advance<bernoulli, 30>::value << '\n';
    std::cout << ctml::advance<bernoulli, 40>::value << '\n';
    std::cout << ctml::advance<bernoulli, 50>::value << '\n';
    std::cout << ctml::advance<bernoulli, 60>::value << '\n';
    std::cout << ctml::advance<bernoulli, 70>::value << '\n';

    return 0;
}
