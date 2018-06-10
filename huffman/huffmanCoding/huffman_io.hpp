#pragma once

#include "bitstream.hpp"
#include "huffman.hpp"

namespace huffman {
    namespace detail {
        template <typename Stream, typename Data, typename Probability, typename Code>
        bitstream::bit_ostream<Stream>& serialize_helper(bitstream::bit_ostream<Stream>& out, const huffman_tree_base<Data, Probability, Code>& root)
        {
            if (root.is_leaf()) {
                out.write_bit(1);
                out << static_cast<const huffman_node<Data, Probability, Code>*>(&root)->data();
              } else {
                out.write_bit(0);
                serialize_helper(out, *(static_cast<const huffman_tree<Data, Probability, Code>*>(&root)->left()));
                serialize_helper(out, *(static_cast<const huffman_tree<Data, Probability, Code>*>(&root)->right()));
            }
            return out;
        }

        template <typename Data, typename Probability, typename Code, typename Stream>
        std::unique_ptr<huffman_tree_base<Data, Probability, Code>> deserialize_helper(bitstream::bit_istream<Stream>& in)
        {
            if (in.read_bit()) {
                Data data;
                in >> data;
                return std::make_unique<huffman_node<Data, Probability, Code>>(std::move(data), Probability{});
            }
            else {
                auto left = deserialize_helper<Data, Probability, Code>(in);
                auto right = deserialize_helper<Data, Probability, Code>(in);
                return std::make_unique<huffman_tree<Data, Probability, Code>>(std::move(left), std::move(right));
            }
        }
    }
    //TODO: use bitstream
    //TODO: add separate method for canonical trees
    //TODO: deserialize
    template <typename Stream, typename Data, typename Probability, typename Code>
    Stream& serialize (Stream& out, const huffman_tree_base<Data, Probability, Code>& root)
    {
        auto wrapped = bitstream::wrap_ostream(out);
        detail::serialize_helper(wrapped, root);
        return out;
    }

    template <typename Data, typename Probability, typename Code, typename Stream>
    std::unique_ptr<huffman_tree<Data, Probability, Code>> deserialize(Stream& in)
    {
        auto wrapped = bitstream::wrap_istream(in);
        return std::unique_ptr<huffman_tree<Data, Probability, Code>>(static_cast<huffman_tree<Data, Probability, Code>*>(detail::deserialize_helper<Data, Probability, Code>(wrapped).release()));
    }
} // namespace huffman
