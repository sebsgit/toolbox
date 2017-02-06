#pragma once

namespace ctml {
    /**
        Compile time pow() with helper methods
    */
    namespace priv {
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
    }

    template <int p, typename T>
    constexpr T pow(const T &t) {
        return priv::pow_base<T, p>::value(t);
    }

    /**
        Random value generated at compile time.
    */
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

    /**
        Normalizes a given value with respect to minimum and maximum values for a given range
        @tparam Range Range type, should expose min() and max() static methods.
        @param value Value to normalize
    */
    template <typename Range>
    constexpr long double normalize(const long double& value) {
        constexpr auto min = Range::min();
        constexpr auto max = Range::max();
        return ((value - min) / (1.0l * (max - min)));
    }
}
