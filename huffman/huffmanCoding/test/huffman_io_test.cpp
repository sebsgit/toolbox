#include "catch.hpp"
#include "huffman_io.hpp"

#include <sstream>

TEST_CASE("huffman_io")
{
    SECTION("basic io")
    {
        std::unordered_map<int, float> probs;
        probs[1] = 0.10f;
        probs[2] = 0.15f;
        probs[3] = 0.30f;
        probs[4] = 0.16f;
        probs[5] = 0.29f;
        using Tree = huffman::huffman_tree<int, float>;
        auto tree = Tree::build(probs.begin(),
            probs.end(),
            [](auto pair) { return pair.first; },
            [](auto pair) { return pair.second; });

        std::stringstream ss;
        huffman::serialize(ss, *tree);
        REQUIRE_FALSE(ss.str().empty());
    }
}