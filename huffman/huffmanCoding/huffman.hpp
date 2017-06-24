#pragma once

#include <vector>
#include <queue>
#include <iostream>
#include <unordered_map>
#include <iterator>
#include <functional>
#include <memory>
#include <cassert>

//TODO:
// serialize / deserialize / canonical form
// configurable "code" type
// optimality api (entropy etc.)
// debugging api (memory usage etc.)
// optimize

template <typename Data, typename Prob, typename Code> class huffman_tree;

template <typename Data, typename Prob, typename Code = std::vector<bool>>
class huffman_tree_base {
public:
    using code_type = Code;

    explicit huffman_tree_base(const Prob& p) : _prob(p) {}
    virtual ~huffman_tree_base() = default;

    virtual Prob prob() const { return this->_prob; }
    virtual Data data() const { throw std::runtime_error("data() called on non-leaf node."); }

    virtual code_type code() const {
        throw std::runtime_error("code() called on non-leaf node.");
        return code_type();
    }

    virtual void prepend_bit(bool bit) = 0;

    virtual bool isLeaf() const { return false; }

    struct greater_than
    {
        bool operator()(const huffman_tree_base* left, const huffman_tree_base* right) const {
            return left->_prob > right->_prob;
        }
    };

    template <typename ForwardIt, typename DataSelector, typename ProbSelector>
    static huffman_tree<Data, Prob, Code>* build(ForwardIt begin, ForwardIt end, const DataSelector& get_data, const ProbSelector& get_prob);

protected:
    Prob _prob;
};

template <typename Data, typename Prob, typename Code = std::vector<bool>>
class huffman_tree : public huffman_tree_base<Data, Prob, Code> {
    using base = huffman_tree_base<Data, Prob, Code>;
public:
    huffman_tree(huffman_tree_base<Data, Prob, Code>* left, huffman_tree_base<Data, Prob, Code>* right)
        :base(left->prob() + right->prob())
        ,_left(left)
        ,_right(right)
    {
        _left->prepend_bit(0);
        _right->prepend_bit(1);
    }

    const base* left() const { return this->_left.get(); }
    const base* right() const { return this->_right.get(); }

    template <typename ForwardIt, typename OutputIt>
    void decode(ForwardIt begin, ForwardIt end, OutputIt out)
    {
        const huffman_tree<Data, Prob, Code>* tree = this;
        for (auto it = begin; it != end; it = std::next(it)) {
            auto next = *it ? tree->right() : tree->left();
            if (!next)
                throw std::runtime_error("invalid code.");
            if (next->isLeaf()) {
                *out = next->data();
                ++out;
                tree = this;
            } else {
                tree = static_cast<const huffman_tree*>(next);
            }
        }
    }

protected:
    void prepend_bit(bool bit) override {
        _left->prepend_bit(bit);
        _right->prepend_bit(bit);
    }

private:
    std::unique_ptr<base> _left;
    std::unique_ptr<base> _right;
};

template <typename Data, typename Prob, typename Code = std::vector<bool>>
class huffman_node : public huffman_tree_base<Data, Prob, Code>
{
    using base = huffman_tree_base<Data, Prob, Code>;
    using code_type = typename base::code_type;
public:
    huffman_node(const Data& d, const Prob& p)
        : base(p)
        , _data(d)
    {
    }

    Data data() const override { return this->_data; }
    code_type code() const override { return this->_code; }

    virtual bool isLeaf() const override { return true; }

protected:
    void prepend_bit(bool bit) override {
        this->_code.insert(this->_code.begin(), bit);
    }

private:
    const Data _data;
    code_type _code;
};


template <typename Data, typename Prob, typename Code> template <class ForwardIt, class DataSelector, class ProbSelector>
huffman_tree<Data, Prob, Code>* huffman_tree_base<Data,Prob, Code>::build(ForwardIt begin, ForwardIt end, const DataSelector& get_data, const ProbSelector& get_prob)
{
    if (begin == end)
        return nullptr;
    using Tree = huffman_tree_base<Data, Prob, Code>;
    std::priority_queue<Tree*, std::vector<Tree*>, Tree::greater_than> nodes;
    for(auto it = begin ; it != end ; it = std::next(it)) {
        nodes.push(new huffman_node<Data, Prob, Code>(get_data(*it), get_prob(*it)));
    }
    while (nodes.size() > 1) {
        Tree* left = nodes.top();
        nodes.pop();
        Tree* right = nodes.top();
        nodes.pop();
        nodes.push(new huffman_tree<Data, Prob, Code>(left, right));
    }
    return dynamic_cast<huffman_tree<Data, Prob, Code>*>(nodes.top());
}

template <typename Data, typename Prob, typename Code = std::vector<bool>>
class huffman_encoder {
    using tree_type = huffman_tree_base<Data, Prob, Code>;
public:
    using code_type = typename tree_type::code_type;

    explicit huffman_encoder(const tree_type* tree)
    {
        assert(tree);
        std::deque<const tree_type*> to_visit;
        to_visit.push_back(tree);
        while (!to_visit.empty()) {
            auto node = to_visit.back();
            to_visit.pop_back();
            if (node->isLeaf()) {
                this->_cache[node->data()] = node->code();
            } else if(auto p = dynamic_cast<const huffman_tree<Data, Prob, Code>*>(node)) {
                if (p->right())
                    to_visit.push_back(p->right());
                if (p->left())
                    to_visit.push_back(p->left());
            }
        }
    }
    code_type code(const Data& data) const { return this->_cache.at(data); }
private:
    std::unordered_map<Data, code_type> _cache;
};
