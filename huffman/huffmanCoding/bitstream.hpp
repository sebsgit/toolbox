#pragma once

#include <cstdint>
#include <exception>
#include <type_traits>
#include <vector>

#if __cpp_if_constexpr >= 201606
#define if_constexpr if constexpr
#else
#define if_constexpr if
#endif

namespace bitstream {
template <typename Stream>
class bit_stream_base {
public:
    explicit bit_stream_base(Stream& s) noexcept
        : _ss(s)
    {
    }
    virtual ~bit_stream_base() = default;

    bool good() const { return this->_ss.good(); }
    bool eof() const { return this->_ss.eof(); }
    bool fail() const { return this->_ss.fail(); }
    bool bad() const { return this->_ss.bad(); }

protected:
    constexpr size_t buffer_size() const noexcept { return sizeof(_buffer) * 8; }

    constexpr static bool is_little_endian() noexcept
    {
        const int16_t a = 1;
        return *reinterpret_cast<const int8_t*>(&a) == 1;
    }

protected:
    Stream& _ss;
    uint8_t _buffer = 0;
    uint8_t _count = 0;
};

template <typename Stream>
class bit_ostream : public bit_stream_base<Stream> {
public:
    explicit bit_ostream(Stream& s) noexcept
        : bit_stream_base<Stream>(s)
    {
    }
    ~bit_ostream()
    {
        try {
            this->flush();
        } catch (...) {
            //TODO:...
        }
    }
    template <typename T>
    bit_ostream& operator<<(const T& t)
    {
        static_assert(std::is_pod<T>::value, "cannot output non-pod type");
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&t);
        if_constexpr (this->is_little_endian()) 
            for (size_t i = 0; i < sizeof(t); ++i) 
                this->write_byte(bytes[i]);
        else 
            for (int32_t i = sizeof(t) - 1; i >= 0; --i) 
                this->write_byte(bytes[i]);
        return *this;
    }
    bit_ostream& operator<<(const std::vector<bool>& data)
    {
        for (auto b : data)
            this->write_bit(b);
        return *this;
    }
    void write_bit(bool bit)
    {
        this->set_bit(static_cast<int>(this->buffer_size() - this->_count - 1), bit);
        ++this->_count;
        if (this->_count == this->buffer_size()) {
            this->_ss << this->_buffer;
            this->_buffer = 0;
            this->_count = 0;
        }
    }
    void write_byte(uint8_t byte)
    {
        for (int8_t b = 7; b >= 0; --b) {
            this->write_bit(byte & (1 << b));
        }
    }
    void flush()
    {
        if (this->_count > 0)
            this->_ss << this->_buffer;
    }

private:
    void set_bit(int bit_num, bool bit)
    {
        this->_buffer = bit ? (this->_buffer | (1 << bit_num)) : (this->_buffer & ~(1 << bit_num));
    }
};

template <typename Stream>
class bit_istream : public bit_stream_base<Stream> {
public:
    explicit bit_istream(Stream& s)
        : bit_stream_base<Stream>(s)
    {
    }
    template <typename T>
    bit_istream& operator>>(T& t)
    {
        static_assert(std::is_pod<T>::value, "cannot write to non-pod type");
        uint8_t* bytes = reinterpret_cast<uint8_t*>(&t);
        if_constexpr (this->is_little_endian())
            for (size_t i = 0; i < sizeof(t); ++i) {
                bytes[i] = this->read_byte();
                if (this->eof())
                    break;
            }
        else
            for (int32_t i = sizeof(t) - 1; i >= 0; --i) {
                bytes[i] = this->read_byte();
                if (this->eof())
                    break;
            }
        return *this;
    }
    bit_istream& operator>>(std::vector<bool>& data)
    {
        while (true) {
            bool bit = this->read_bit();
            if (this->eof())
                break;
            data.push_back(bit);
        }
        return *this;
    }
    bool read_bit()
    {
        if (this->eof())
            throw std::runtime_error("cannot read, EOF reached.");
        if (this->_count == 0) {
            this->_ss >> this->_buffer;
            this->_count = static_cast<uint8_t>(this->buffer_size());
        }
        bool result = this->get_bit(this->_count - 1);
        --this->_count;
        return result;
    }
    uint8_t read_byte()
    {
        uint8_t result = 0;
        for (int8_t b = 7; b >= 0; --b) {
            bool bit = this->read_bit();
            if (this->eof())
                break;
            result = bit ? (result | (1 << b)) : (result & ~(1 << b));
        }
        return result;
    }
    std::vector<bool> read_bits(size_t count)
    {
        std::vector<bool> result;
        result.reserve(count);
        this->read_bits(count, std::back_inserter(result));
        return result;
    }
    template <typename OutputIt>
    void read_bits(size_t count, OutputIt out)
    {
        while (count > 0) {
            bool bit = this->read_bit();
            if (this->eof())
                break;
            *out = bit;
            ++out;
            --count;
        }
    }
    void skip(size_t count)
    {
        struct ignore_iterator {
            constexpr auto& operator*() const noexcept { return std::ignore; }
            constexpr auto& operator++() const noexcept { return *this; }
        };
        this->read_bits(count, ignore_iterator());
    }

private:
    bool get_bit(int bit_num) const noexcept
    {
        return (this->_buffer & (1 << bit_num)) != 0;
    }
};

template <typename U>
bit_ostream<U> wrap_ostream(U& stream)
{
    return bit_ostream<U>(stream);
}

template <typename U>
bit_istream<U> wrap_istream(U& stream)
{
    return bit_istream<U>(stream);
}
} // namespace bitstream

#undef if_constexpr
