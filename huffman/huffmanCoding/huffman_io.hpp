#pragma once

#include "bitstream.hpp"
#include "huffman.hpp"

namespace huffman {
    //TODO: use bitstream
    //TODO: add separate method for canonical trees
    //TODO: deserialize
    template <typename Stream, typename Data, typename Probability, typename Code>
    Stream& serialize (Stream& out, const huffman_tree_base<Data, Probability, Code>& root)
    {
        if (root.is_leaf()) {
            out << 1 << static_cast<const huffman_node<Data, Probability, Code>*>(&root)->data();
        }
        else {
            out << 0;
            serialize(out, *(static_cast<const huffman_tree<Data, Probability, Code>*>(&root)->left()));
            serialize(out, *(static_cast<const huffman_tree<Data, Probability, Code>*>(&root)->right()));
        }
        return out;
    }
} // namespace huffman
