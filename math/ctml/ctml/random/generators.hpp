#pragma once

#include "ctml/utils.hpp"

namespace ctml {
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
}
