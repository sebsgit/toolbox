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
    static_assert(std::is_same<int, one::type>::value);
    static_assert(one::value == 1);
    static_assert(one::next::value == 2);
    static_assert(one::next::next::value == 3);
    static_assert(one::next::next::next::value == 4);
    static_assert(ctml::advance<ctml::sequence_iterator<plus_one>, 100>::value == 100);
    static_assert(ctml::accumulate<ctml::sequence_iterator<plus_one>, 2>::value == 3, "0 + 1 + 2");
    static_assert(ctml::accumulate<ctml::sequence_iterator<plus_one>, 10>::value == 55, "0 + 1 + 2 + ... + 10");
    static_assert(ctml::mean<ctml::sequence_iterator<plus_one>, 10>::value == 5.5, "55 / 10");

    class times_two {
    public:
        constexpr unsigned operator() (int v) const {
            return v * 2;
        }
    };
    using one_1 = ctml::sequence_iterator<times_two, 1>;
    static_assert(std::is_same<unsigned, one_1::type>::value);
    static_assert(one_1::value == 1);
    static_assert(one_1::next::value == 2);
    static_assert(one_1::next::next::value == 4);
    static_assert(one_1::next::next::next::value == 8);
    static_assert(ctml::advance<one_1, 0>::value == 1);
    static_assert(ctml::advance<one_1, 10>::value == 1024);

    class div_by_10 {
    public:
        constexpr short operator() (short v) const {
            return v / 10.0;
        }
    };
    using thousand = ctml::sequence_iterator<div_by_10, 1000>;
    static_assert(std::is_same<short, thousand::type>::value);
    static_assert(thousand::value == 1000);
    static_assert(thousand::next::value == 100);
    static_assert(thousand::next::next::value == 10);
    static_assert(thousand::next::next::next::value == 1);
}

template <typename Dist, int steps>
void testDistribution(const std::string& name)
{
	std::cout << name << '\n';
    std::cout << ctml::advance<Dist, 10>::value << ' ';
    std::cout << ctml::advance<Dist, 20>::value << ' ';
    std::cout << ctml::advance<Dist, 30>::value << ' ';
    std::cout << ctml::advance<Dist, 40>::value << ' ';
    std::cout << ctml::advance<Dist, 50>::value << ' ';
    std::cout << ctml::advance<Dist, 60>::value << ' ';
    std::cout << ctml::advance<Dist, 70>::value << '\n';
    std::cout << "mean: " << ctml::mean<Dist, steps>::value << ", ideal: " << Dist::mean <<  '\n';
}

int main(int argc, char *argv[]) {
    static_assert(ctml::pow<2>(2) == 4);
    static_assert(ctml::pow<3>(2.2) == 2.2 * 2.2 * 2.2);
    static_assert(ctml::pow<0>(5.2) == 1);
    static_assert(ctml::pow<-2>(2.2) == 1.0 / (2.2 * 2.2));
    
    // std::pow not a constexpr in clang 3.8
    #ifndef __clang__
    static_assert(std::pow(2, 2) == 4);
    static_assert(std::pow(2.2, -1) == 1.0 / (2.2));
    #endif
    
    static_assert(ctml::random_generators::blum_blum_shub<int>()(2) > 0, "bbs");
    static_assert(ctml::random_generators::linear_congruential<int, 2, 3, 5>()(2) == 2, "LCG");
    testSequenceIterator();

    std::cout << "bbs:\n";
    using bbs_seq = ctml::sequence_iterator<ctml::random_generators::blum_blum_shub<unsigned long long>, 23843872834>;
    std::cout << bbs_seq::value << ' ';
    std::cout << ctml::advance<bbs_seq, 1>::value << ' ';
    std::cout << ctml::advance<bbs_seq, 2>::value << ' ';
    std::cout << ctml::advance<bbs_seq, 3>::value << ' ';
    std::cout << ctml::advance<bbs_seq, 4>::value << ' ';
    std::cout << ctml::advance<bbs_seq, 5>::value << ' ';
    std::cout << ctml::advance<bbs_seq, 6>::value << ' ';
    std::cout << ctml::advance<bbs_seq, 7>::value << '\n';

	constexpr int steps = 350;
    struct uniform_params {
        static constexpr double a() { return 0.0; }
        static constexpr double b() { return 1.0; }
    };
    using unigen = ctml::random_distributions::uniform<ctml::random_generators::blum_blum_shub<unsigned long long>, uniform_params>;
    testDistribution<unigen, steps>("uniform");

    using unigen_disc = ctml::random_distributions::uniform_discrete<ctml::random_generators::blum_blum_shub<unsigned long long>, 1, 1000>;
    testDistribution<unigen_disc, steps>("uniform discrete");

    struct bernoulli_params { static constexpr double p() { return 0.32; } };
    using bernoulli = ctml::random_distributions::bernoulli<ctml::random_generators::linear_congruential<unsigned long long, 13, 92, 43>, bernoulli_params>;
    testDistribution<bernoulli, steps>("bernoulli");

    return 0;
}
