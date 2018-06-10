#include "catch.hpp"
#include "huffman_io.hpp"

#include <fstream>

TEST_CASE("huffman_io")
{
    SECTION("basic io")
    {
        std::unordered_map<int, std::vector<bool>> codes;
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

            for (const auto& leaf : *tree)
                codes[leaf.data()] = leaf.code();
            REQUIRE(codes.size() == 5);

            std::ofstream out("test.dat", std::ios_base::out | std::ios_base::binary);
            huffman::serialize(out, *tree);
            out.close();
        }

        std::ifstream in("test.dat", std::ios_base::out | std::ios_base::binary);
        in.seekg(std::ios_base::end);
        int size = static_cast<int>(in.tellg());
        REQUIRE(size < 2 * 5 * sizeof(int));
        in.seekg(0, std::ios_base::beg);
        auto tree2 = huffman::deserialize<int, float, std::vector<bool>>(in);
        REQUIRE(tree2);
        std::remove("test.dat");
        std::unordered_map<int, std::vector<bool>> codes2;
        for (const auto& leaf : *tree2) {
            codes2[leaf.data()] = leaf.code();
        }
        REQUIRE(codes2.size() == 5);
        for (const auto& data : codes2) {
            REQUIRE(codes[data.first] == data.second);
        }
    }
}