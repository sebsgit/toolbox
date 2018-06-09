#pragma once

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <vector>

//TODO:
// refactoring, create namespace, cleanups
// more constexpr
// serialize / deserialize / canonical form
// configurable "code" type
// optimality api (entropy etc.)
// debugging api (memory usage etc.)
// optimize

template <typename Data, typename Prob, typename Code>
class huffman_tree;

template <typename Data, typename Prob, typename Code = std::vector<bool>>
class huffman_tree_base {
public:
    using code_type = Code;

    // iterator
    //TODO: don't inherit std::iterator
    template <typename _Tp>
    class iterator_base : public std::iterator<std::forward_iterator_tag, _Tp> {
    public:
        explicit iterator_base(_Tp* current)
            : _current(current)
        {
            if (current) {
                this->_to_visit.push_back(current);
                // position on the first leaf
                while (this->_current && !this->_current->is_leaf()) {
                    this->operator++();
                }
            }
        }

        iterator_base& operator++()
        {
            using tree_type = typename std::conditional<std::is_const<_Tp>::value, const huffman_tree<Data, Prob, Code>, huffman_tree<Data, Prob, Code>>::type;
            if (this->_to_visit.empty())
                this->_current = nullptr;
            while (!this->_to_visit.empty()) {
                auto node = this->_to_visit.back();
                this->_to_visit.pop_back();
                if (node->is_leaf()) {
                    this->_current = node;
                    break;
                } else if (auto p = dynamic_cast<tree_type*>(node)) {
                    if (p->right())
                        this->_to_visit.push_back(p->right());
                    if (p->left())
                        this->_to_visit.push_back(p->left());
                }
            }
            return *this;
        }
        const huffman_tree_base& operator*() const { return *this->_current; }

        bool operator==(const iterator_base& other) const { return this->_current == other._current; }
        bool operator!=(const iterator_base& other) const { return !(*this == other); }

    private:
        _Tp* _current;
        std::deque<_Tp*> _to_visit;
    };

    using const_iterator = iterator_base<const huffman_tree_base>;
    using iterator = iterator_base<huffman_tree_base>;
    //

    explicit huffman_tree_base(const Prob& p)
        : _prob(p)
    {
    }
    virtual ~huffman_tree_base() = default;

    //TODO: spelling
    virtual Prob prob() const { return this->_prob; }
    virtual Data data() const { throw std::runtime_error("data() called on non-leaf node."); }

    virtual code_type code() const
    {
        throw std::runtime_error("code() called on non-leaf node.");
        return code_type();
    }
    virtual size_t code_length() const
    {
        throw std::runtime_error("code() called on non-leaf node.");
        return size_t();
    }

    virtual void prepend_bit(bool bit) = 0;

    virtual bool is_leaf() const { return false; }

    virtual void make_canonical() {}

    struct greater_than {
        bool operator()(const huffman_tree_base* left, const huffman_tree_base* right) const
        {
            return left->_prob > right->_prob;
        }
    };

    const_iterator begin() const { return const_iterator(this); }
    const_iterator end() const { return const_iterator(nullptr); }

    iterator begin() { return iterator(this); }
    iterator end() { return iterator(nullptr); }

    template <typename ForwardIt, typename DataSelector, typename ProbSelector>
    static std::unique_ptr<huffman_tree<Data, Prob, Code>> build(ForwardIt begin, ForwardIt end, const DataSelector& get_data, const ProbSelector& get_prob);

protected:
    virtual void invalidate_probabilities() {}

protected:
    Prob _prob;
};

template <typename Data, typename Prob, typename Code = std::vector<bool>>
class huffman_tree : public huffman_tree_base<Data, Prob, Code> {
    using base = huffman_tree_base<Data, Prob, Code>;

public:
    huffman_tree(huffman_tree_base<Data, Prob, Code>* left, huffman_tree_base<Data, Prob, Code>* right)
        : base(left->prob() + right->prob())
        , _left(left)
        , _right(right)
    {
        _left->prepend_bit(0);
        _right->prepend_bit(1);
    }

    const base* left() const { return this->_left.get(); }
    const base* right() const { return this->_right.get(); }

    base* left() { return this->_left.get(); }
    base* right() { return this->_right.get(); }

    template <typename ForwardIt, typename OutputIt>
    void decode(ForwardIt begin, ForwardIt end, OutputIt out)
    {
        const huffman_tree<Data, Prob, Code>* tree = this;
        for (auto it = begin; it != end; it = std::next(it)) {
            auto next = *it ? tree->right() : tree->left();
            if (!next)
                throw std::runtime_error("invalid code.");
            if (next->is_leaf()) {
                *out = next->data();
                ++out;
                tree = this;
            } else {
                tree = static_cast<const huffman_tree*>(next);
            }
        }
    }

    void make_canonical() override;

protected:
    huffman_tree()
        : base(Prob())
    {
    }

    void prepend_bit(bool bit) override
    {
        _left->prepend_bit(bit);
        _right->prepend_bit(bit);
    }

    base* take_left() { return this->_left.release(); }
    base* take_right() { return this->_right.release(); }

    void push_node(base* node)
    {
        assert(node);
        assert(node->code_length() > 0);
        const auto code = node->code();
        huffman_tree* current = this;
        for (size_t i = 0; i < node->code_length() - 1; ++i) {
            bool to_left = (code[i] == false);
            auto& to_insert = (to_left ? current->_left : current->_right);
            if (!to_insert)
                to_insert.reset(new huffman_tree());
            current = dynamic_cast<huffman_tree*>(to_insert.get());
        }
        auto& to_insert = (code[node->code_length() - 1] ? current->_right : current->_left);
        if (to_insert.get())
            throw std::runtime_error("node allocated"); //TODO: more descriptive message, trigger with unit test
        to_insert.reset(node);
    }
    void invalidate_probabilities() override
    {
        this->_prob = (this->left() ? this->left()->prob() : Prob()) + (this->right() ? this->right()->prob() : Prob());
    }

    static std::string to_string(const std::vector<bool>& v)
    {
        std::stringstream ss;
        std::copy(v.begin(), v.end(), std::ostream_iterator<bool>(ss, ""));
        return ss.str();
    }
    static std::vector<bool> zeros(size_t length) { return std::vector<bool>(length, false); }
    static void fill_zeros(std::vector<bool>& v, size_t new_length)
    {
        if (v.size() < new_length) {
            v.insert(v.end(), new_length - v.size(), false);
        }
    }
    static void increment_binary(std::vector<bool>& v)
    {
        assert(!v.empty()); //TODO: something better than assert()
        int pos = static_cast<int>(v.size()) - 1;
        while (pos >= 0 && v[pos] == true) {
            v[pos] = false;
            --pos;
        }
        if (pos >= 0)
            v[pos] = true;
        else
            v.insert(v.begin(), true);
    }

private:
    std::unique_ptr<base> _left;
    std::unique_ptr<base> _right;
};

template <typename Data, typename Prob, typename Code = std::vector<bool>>
class huffman_node : public huffman_tree_base<Data, Prob, Code> {
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
    size_t code_length() const override { return this->_code.size(); }

    void set_code(const code_type& new_code) { this->_code = new_code; }

    virtual bool is_leaf() const override { return true; }

protected:
    void prepend_bit(bool bit) override
    {
        this->_code.insert(this->_code.begin(), bit);
    }

private:
    const Data _data;
    code_type _code; //TODO: keep consisten naming
};

template <typename Data, typename Prob, typename Code>
void huffman_tree<Data, Prob, Code>::make_canonical()
{
    if (this->is_leaf())
        return;
    using node = huffman_node<Data, Prob, Code>;
    // detach all leaf nodes
    std::vector<node*> nodes;
    std::deque<huffman_tree*> to_check;
    to_check.push_back(this);
    while (!to_check.empty()) {
        auto parent = to_check.back();
        to_check.pop_back();
        if (parent->left()) {
            if (parent->left()->is_leaf())
                nodes.push_back(dynamic_cast<node*>(parent->take_left()));
            else
                to_check.push_back(dynamic_cast<huffman_tree*>(parent->left()));
        }
        if (parent->right()) {
            if (parent->right()->is_leaf())
                nodes.push_back(dynamic_cast<node*>(parent->take_right()));
            else
                to_check.push_back(dynamic_cast<huffman_tree*>(parent->right()));
        }
    }
    if (!nodes.empty()) {
        this->_left.reset();
        this->_right.reset();
        // sort leaf nodes by code length and probability
        std::sort(nodes.begin(), nodes.end(), [](const node* left, const node* right) {
            return left->code_length() == right->code_length() ? left->prob() > right->prob() : left->code_length() < right->code_length();
        });
        // assign new canonical codes
        size_t last_length = nodes[0]->code_length();
        Code to_set = zeros(last_length);
        for (auto& n : nodes) {
            if (n->code_length() != last_length) {
                last_length = n->code_length();
                fill_zeros(to_set, last_length);
            }
            n->set_code(to_set);
            // insert the new node
            this->push_node(n);
            increment_binary(to_set);
        }
        // finally rebuild the probabilites
        this->invalidate_probabilities();
    }
}

template <typename Data, typename Prob, typename Code>
template <class ForwardIt, class DataSelector, class ProbSelector>
std::unique_ptr<huffman_tree<Data, Prob, Code>> huffman_tree_base<Data, Prob, Code>::build(ForwardIt begin, ForwardIt end, const DataSelector& get_data, const ProbSelector& get_prob)
{
    if (begin == end)
        return nullptr;
    using Tree = huffman_tree_base<Data, Prob, Code>;
    std::priority_queue<Tree*, std::vector<Tree*>, Tree::greater_than> nodes;
    for (auto it = begin; it != end; it = std::next(it)) {
        nodes.push(new huffman_node<Data, Prob, Code>(get_data(*it), get_prob(*it)));
    }
    assert(!nodes.empty());
    while (nodes.size() > 1) {
        Tree* left = nodes.top();
        nodes.pop();
        Tree* right = nodes.top();
        nodes.pop();
        nodes.push(new huffman_tree<Data, Prob, Code>(left, right));
    }
    return std::unique_ptr<huffman_tree<Data, Prob, Code>>(dynamic_cast<huffman_tree<Data, Prob, Code>*>(nodes.top()));
}

template <typename Data, typename Prob, typename Code = std::vector<bool>>
class huffman_encoder {
    using tree_type = huffman_tree_base<Data, Prob, Code>;

public:
    using code_type = typename tree_type::code_type;

    explicit huffman_encoder(const tree_type& tree)
    {
        for (const auto& it : tree)
            this->_cache[it.data()] = it.code();
    }
    code_type code(const Data& data) const { return this->_cache.at(data); }

private:
    std::unordered_map<Data, code_type> _cache;
};

template <typename Data, typename Prob, typename Code>
auto make_encoder(const huffman_tree_base<Data, Prob, Code>& tree)
{
    return huffman_encoder<Data, Prob, Code>(tree);
}
