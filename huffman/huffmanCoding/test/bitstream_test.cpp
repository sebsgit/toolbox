#include "catch.hpp"
#include "bitstream.hpp"

#include <vector>
#include <fstream>
#include <sstream>

TEST_CASE("bitstream")
{
    SECTION("simple")
    {
        std::stringstream ss;
        auto out = bitstream::wrap_ostream(ss);
        out.write_bit(1);
        out << 0.45f;
        out.write_bit(1);
        out.write_bit(0);
        out.write_bit(1);
        out << 1.9f;
        out.write_bit(1);
        out.flush();
        auto s = ss.str();
        REQUIRE_FALSE(s.empty());
        ss = std::stringstream(s);
        auto in = bitstream::wrap_istream(ss);
        REQUIRE(in.read_bit());
        float tmp = 0.0f;
        in >> tmp;
        REQUIRE(tmp == 0.45f);
        REQUIRE(in.read_bits(3) == std::vector<bool>{1, 0, 1});
        in >> tmp;
        REQUIRE(tmp == 1.9f);
        REQUIRE(in.read_bit());
    }
    SECTION("file I/O")
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
        REQUIRE_FALSE(is.eof());
        is >> x;
        is >> a;
        is >> b;
        REQUIRE(x == 45);
        REQUIRE(a == 37);
        REQUIRE(b == 3);
        in.seekg(0);
        is.skip(sizeof(x) * 8);
        std::vector<bool> read_bits = is.read_bits(8);
        REQUIRE(read_bits == data);
        read_bits.clear();
        is >> read_bits;
        REQUIRE(is.eof());
        REQUIRE(read_bits.size() == 8);
        REQUIRE(read_bits[0] == false);
        REQUIRE(read_bits[1] == false);
        REQUIRE(read_bits[2] == false);
        REQUIRE(read_bits[3] == false);
        REQUIRE(read_bits[4] == false);
        REQUIRE(read_bits[5] == false);
        REQUIRE(read_bits[6] == true);
        REQUIRE(read_bits[7] == true);

        std::remove("test.dat");
    }
}
