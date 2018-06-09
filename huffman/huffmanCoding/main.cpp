#include "bitstream.hpp"
#include "huffman.hpp"

#include <fstream>

static void test_huffman()
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

    using Tree = huffman_tree_base<int, float>;
    auto tree = Tree::build(probs.begin(),
        probs.end(),
        [](auto pair) { return pair.first; },
        [](auto pair) { return pair.second; });
    auto encoder = make_encoder(*tree);
    assert(encoder.code(2) == std::vector<bool>({ 0, 1, 1 }));
    for (const auto& p : expected_codes) {
        assert(encoder.code(p.first) == p.second);
    }
    std::vector<bool> code{ 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0 };
    std::vector<int> message;
    tree->decode(code.begin(), code.end(), std::back_inserter(message));
    assert(message.size() == 5);
    assert(message == std::vector<int>({ 1, 2, 4, 4, 5 }));
}

static void test_huffman_canonical_form()
{
    std::unordered_map<char, double> probs;
    probs['B'] = 0.4;
    probs['A'] = 0.3;
    probs['C'] = 0.2;
    probs['D'] = 0.1;
    using Tree = huffman_tree_base<char, double>;
    auto tree = Tree::build(probs.begin(),
        probs.end(),
        [](auto pair) { return pair.first; },
        [](auto pair) { return pair.second; });
    tree->make_canonical();
    auto encoder = make_encoder(*tree);
    assert(encoder.code('B') == std::vector<bool>({ 0 }));
    assert(encoder.code('A') == std::vector<bool>({ 1, 0 }));
    assert(encoder.code('C') == std::vector<bool>({ 1, 1, 0 }));
    assert(encoder.code('D') == std::vector<bool>({ 1, 1, 1 }));
}

static void test_bitstream()
{
    std::vector<bool> data{ 0, 0, 1, 0, 0, 1, 0, 1 }; // 37
    std::ofstream out("test.dat", std::ios_base::out | std::ios_base::binary);
    auto ss = bitstream::wrap_ostream(out);
    int16_t x = 45;
    ss << x;
    ss << data;
    ss.write_bit(0);
    ss.write_bit(0);
    ss.write_bit(0);
    ss.write_bit(0);

    ss.write_bit(0);
    ss.write_bit(0);
    ss.write_bit(1);
    ss.write_bit(1);
    out.close();
    std::ifstream in("test.dat", std::ios_base::in | std::ios_base::binary);
    uint8_t a, b;
    auto is = bitstream::wrap_istream(in);
    assert(!is.eof());
    is >> x;
    is >> a;
    is >> b;
    assert(x == 45);
    assert(a == 37);
    assert(b == 3);
    in.seekg(0);
    is.skip(sizeof(x) * 8);
    std::vector<bool> read_bits = is.read_bits(8);
    assert(read_bits == data);
    read_bits.clear();
    is >> read_bits;
    assert(is.eof());
    assert(read_bits.size() == 8);
    assert(read_bits[0] == false);
    assert(read_bits[1] == false);
    assert(read_bits[2] == false);
    assert(read_bits[3] == false);
    assert(read_bits[4] == false);
    assert(read_bits[5] == false);
    assert(read_bits[6] == true);
    assert(read_bits[7] == true);

    std::remove("test.dat");
}

int main()
{
    test_huffman();
    test_huffman_canonical_form();
    test_bitstream();
    return 0;
}
