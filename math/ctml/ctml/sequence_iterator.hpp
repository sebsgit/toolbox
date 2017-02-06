#pragma once

namespace ctml {
    /**
        Compile time sequence iterator
        @tparam Gen Generator type. Should support operator()(seed)
        @tparam seed Number to start the sequence with. Defaults to Generator(0)
    */
    template <typename Gen, decltype(Gen().operator()(0)) seed = decltype(Gen().operator()(0))()>
    struct sequence_iterator {
        using generator = Gen;
        using type = decltype(seed);
        static constexpr type value = seed;
        using next = sequence_iterator<Gen, Gen()(seed)>;
    };

    /**
        Advances the sequence iterator by a given number of steps
    */
    template <typename SequenceIterator, int steps> struct advance;
    template <typename SequenceIterator>
    struct advance<SequenceIterator, 0> {
        static constexpr auto value = SequenceIterator::value;
    };
    template <typename SequenceIterator, int steps>
    struct advance {
        static constexpr auto value = advance<typename SequenceIterator::next, steps - 1>::value;
    };

    /**
        Accumulates a given number of sequence values
    */
    template <typename SequenceIterator, int steps> struct accumulate;
    template <typename SequenceIterator>
    struct accumulate<SequenceIterator, 0> {
        static constexpr auto value = SequenceIterator::value;
    };
    template <typename SequenceIterator, int steps>
    struct accumulate {
        static constexpr auto value = accumulate<typename SequenceIterator::next, steps - 1>::value + SequenceIterator::value;
    };
}
