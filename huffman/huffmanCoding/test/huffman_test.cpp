#include "catch.hpp"
#include "huffman.hpp"

#include <unordered_map>
#include <vector>

TEST_CASE("huffman")
{
    SECTION("basic")
    {
        std::unordered_map<int, float> probs;
        probs[1] = 0.10f;
        probs[2] = 0.15f;
        probs[3] = 0.30f;
        probs[4] = 0.16f;
        probs[5] = 0.29f;
        std::unordered_map<int, std::vector<bool>> expected_codes;
        expected_codes[1] = { 0, 1, 0 };
        expected_codes[2] = { 0, 1, 1 };
        expected_codes[3] = { 1, 1 };
        expected_codes[4] = { 0, 0 };
        expected_codes[5] = { 1, 0 };

        using Tree = huffman::huffman_tree<int, float>;
        auto tree = Tree::build(probs.begin(),
            probs.end(),
            [](auto pair) { return pair.first; },
            [](auto pair) { return pair.second; });
        auto encoder = make_encoder(*tree);
        REQUIRE(encoder.code(2) == std::vector<bool>({ 0, 1, 1 }));
        for (const auto& p : expected_codes) {
            REQUIRE(encoder.code(p.first) == p.second);
        }
        std::vector<bool> code{ 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0 };
        std::vector<int> message;
        auto res = tree->decode(code.begin(), code.end(), std::back_inserter(message));
        REQUIRE(res == Tree::decode_result::ok);
        REQUIRE(message.size() == 5);
        REQUIRE(message == std::vector<int>({ 1, 2, 4, 4, 5 }));
        code = std::vector<bool>{0, 1, 1, 1};
        res = tree->decode(code.begin(), code.end(), std::back_inserter(message));
        REQUIRE(res == Tree::decode_result::invalid_code);
    }

    SECTION("canonical form")
    {
        std::unordered_map<char, double> probs;
        probs['B'] = 0.4;
        probs['A'] = 0.3;
        probs['C'] = 0.2;
        probs['D'] = 0.1;
        using Tree = huffman::huffman_tree_base<char, double>;
        auto tree = Tree::build(probs.begin(),
            probs.end(),
            [](auto pair) { return pair.first; },
            [](auto pair) { return pair.second; });
        huffman::make_canonical(*tree);
        auto encoder = make_encoder(*tree);
        REQUIRE(encoder.code('B') == std::vector<bool>({ 0 }));
        REQUIRE(encoder.code('A') == std::vector<bool>({ 1, 0 }));
        REQUIRE(encoder.code('C') == std::vector<bool>({ 1, 1, 0 }));
        REQUIRE(encoder.code('D') == std::vector<bool>({ 1, 1, 1 }));
    }
}