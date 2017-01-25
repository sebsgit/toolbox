#include <cmath>
#include <iostream>
#include <type_traits>

namespace ctml {
template <typename T, int p, bool isPositive = (p>0) > class pow_base;

template <typename T>
class pow_base<T, 0> {
public:
    static constexpr T value(const T &) { return 1; }
};
template <typename T, int p>
class pow_base<T, p, true> {
public:
    static constexpr T value(const T &t) {
        return t * pow_base<T, p - 1>::value(t);
    }
};
template <typename T, int p>
class pow_base<T, p, false> {
public:
    static constexpr double value(const T &t) {
        return 1.0 / (pow_base<T, -p>::value(t));
    }
};
template <int p, typename T>
constexpr T pow(const T &t) {
    return pow_base<T, p>::value(t);
}

template <typename Gen, decltype(Gen().operator()(0)) seed = decltype(Gen().operator()(0))()>
struct sequence_iterator {
    using generator = Gen;
    using type = decltype(seed);
    static constexpr type value = seed;
    using next = sequence_iterator<Gen, Gen()(seed)>;
};
template <typename SequenceIterator, int steps> struct advance;
template <typename SequenceIterator>
struct advance<SequenceIterator, 0> {
    static constexpr auto value = SequenceIterator::value;
};
template <typename SequenceIterator, int steps>
struct advance
{
    static constexpr auto value = advance<typename SequenceIterator::next, steps - 1>::value;
};

namespace random_generators {
    template <typename T, unsigned long p = 20359, unsigned long q = 43063>
    class blum_blum_shub {
    public:
        using type = T;
        constexpr T operator()(const T &prev) const { return pow<2>(prev) % (p * q); }
        static constexpr T max() { return p * q - 1; }
        static constexpr T min() { return 0; }
    };
    template <typename T, int a, int c, int m>
    class linear_congruential {
    public:
        using type = T;
        constexpr T operator()(const T &x) const { return (a * x + c) % m; }
        static constexpr T max() { return m - 1; }
        static constexpr T min() { return 0; }
    };
}

template <typename T>
class random_seed {
private:
    static constexpr T value_helper() {
        constexpr char t[] = __TIME__;
        return (t[0] - '0') * 100000 + (t[1] - '0') * 10000 + (t[3] - '0') * 1000  + (t[4] - '0') + ((t[6] - '0') % 100) + (t[7] - '0');
    }
public:
    static constexpr auto value = value_helper();
};

namespace random_distributions {
    template <typename Generator>
    constexpr long double normalize(const typename Generator::type& value) {
        constexpr auto min = Generator::min();
        constexpr auto max = Generator::max();
        return ((value - min) / (1.0l * (max - min)));
    }

    template <typename Generator, long a, long b, uint64_t seed = random_seed<uint64_t>::value, typename SeqInit = typename ctml::sequence_iterator<Generator, seed>::next>
    class uniform {
    private:
        static constexpr typename Generator::type value_helper() {
            static_assert(a <= b, "error: cannot generate uniform distribution with b < a");
            using generator = typename SeqInit::generator;
            return normalize<generator>(SeqInit::value) * (b - a) + a;
        }
    public:
        using next = uniform<Generator, a, b, seed, typename SeqInit::next>;
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

    std::cout << "uniform:\n";
    using unigen = ctml::random_distributions::uniform<ctml::random_generators::blum_blum_shub<unsigned long long>, 1, 100>;
    std::cout << ctml::advance<unigen, 10>::value << '\n';
    std::cout << ctml::advance<unigen, 20>::value << '\n';
    std::cout << ctml::advance<unigen, 30>::value << '\n';
    std::cout << ctml::advance<unigen, 40>::value << '\n';
    std::cout << ctml::advance<unigen, 50>::value << '\n';
    std::cout << ctml::advance<unigen, 60>::value << '\n';
    std::cout << ctml::advance<unigen, 70>::value << '\n';

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
