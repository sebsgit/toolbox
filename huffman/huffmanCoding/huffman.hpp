#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <vector>

//TODO:
// remove helper methods from the tree class
// serialize / deserialize / canonical form
// add build_canonical
// configurable "code" type
// optimality api (entropy etc.)
// debugging api (memory usage etc.)
// make Probability optional in node
// optimize

namespace huffman {

template <typename Data, typename Probability, typename Code>
class huffman_tree;

template <typename Data, typename Probability, typename Code>
class huffman_node;

template <typename Data, typename Probability, typename Code = std::vector<bool>>
class huffman_tree_base {
public:
    using code_type = Code;

    enum class type {
        leaf_node,
        internal_node
    };

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
            using tree_type = typename std::conditional<std::is_const<_Tp>::value, const huffman_tree<Data, Probability, Code>, huffman_tree<Data, Probability, Code>>::type;
            if (this->_to_visit.empty())
                this->_current = nullptr;
            while (!this->_to_visit.empty()) {
                auto node = this->_to_visit.back();
                this->_to_visit.pop_back();
                if (node->is_leaf()) {
                    this->_current = node;
                    break;
                } else {
                    const auto p = static_cast<tree_type*>(node);
                    if (p->right())
                        this->_to_visit.push_back(p->right());
                    if (p->left())
                        this->_to_visit.push_back(p->left());
                }
            }
            return *this;
        }
        const auto& operator*() const { return *static_cast<const huffman_node<Data, Probability, Code>*>(this->_current); }

        bool operator==(const iterator_base& other) const { return this->_current == other._current; }
        bool operator!=(const iterator_base& other) const { return !(*this == other); }

    private:
        _Tp* _current;
        std::deque<_Tp*> _to_visit;
    };

    using const_iterator = iterator_base<const huffman_tree_base>;
    using iterator = iterator_base<huffman_tree_base>;
    //

    explicit huffman_tree_base(const Probability& p, const type t)
        : _probability(p)
        , _type(t)
    {
    }
    virtual ~huffman_tree_base() = default;

    auto probability() const { return this->_probability; }

    virtual void prepend_bit(bool bit) = 0;

    bool is_leaf() const noexcept { return this->_type == type::leaf_node; }

    const_iterator begin() const { return const_iterator(this); }
    const_iterator end() const { return const_iterator(nullptr); }

    iterator begin() { return iterator(this); }
    iterator end() { return iterator(nullptr); }

    template <typename ForwardIt, typename DataSelector, typename ProbSelector>
    static std::unique_ptr<huffman_tree<Data, Probability, Code>> build(ForwardIt begin, ForwardIt end, const DataSelector& get_data, const ProbSelector& get_prob);

protected:
    virtual void invalidate_probabilities() {}

protected:
    Probability _probability;
    const type _type;
};

template <typename Data, typename Probability, typename Code = std::vector<bool>>
class huffman_tree : public huffman_tree_base<Data, Probability, Code> {
    using base = huffman_tree_base<Data, Probability, Code>;
    using leaf = huffman_node<Data, Probability, Code>;

public:
    enum class decode_result {
        ok,
        invalid_code
    };

    huffman_tree(std::unique_ptr<huffman_tree_base<Data, Probability, Code>>&& left, std::unique_ptr<huffman_tree_base<Data, Probability, Code>>&& right)
        : base(left->probability() + right->probability(), base::type::internal_node)
        , _left(std::move(left))
        , _right(std::move(right))
    {
        _left->prepend_bit(0);
        _right->prepend_bit(1);
    }

    const base* left() const { return this->_left.get(); }
    const base* right() const { return this->_right.get(); }

    base* left() { return this->_left.get(); }
    base* right() { return this->_right.get(); }

    template <typename ForwardIt, typename OutputIt>
    decode_result decode(ForwardIt begin, ForwardIt end, OutputIt out)
    {
        const huffman_tree<Data, Probability, Code>* tree = this;
        for (auto it = begin; it != end; it = std::next(it)) {
            auto next = *it ? tree->right() : tree->left();
            if (!next)
                return decode_result::invalid_code;
            if (next->is_leaf()) {
                *out = static_cast<const leaf*>(next)->data();
                ++out;
                tree = this;
            } else {
                tree = static_cast<const huffman_tree*>(next);
            }
        }
        return tree == this ? decode_result::ok : decode_result::invalid_code;
    }

    void reset()
    {
        this->_left.reset();
        this->_right.reset();
    }

    bool push_node(leaf* node)
    {
        if (!node || node->code_length() == 0)
            return false;
        const auto code = node->code();
        huffman_tree* current = this;
        for (size_t i = 0; i < node->code_length() - 1; ++i) {
            bool to_left = (code[i] == false);
            auto& to_insert = (to_left ? current->_left : current->_right);
            if (!to_insert || to_insert->is_leaf())
                to_insert.reset(new huffman_tree());
            current = static_cast<huffman_tree*>(to_insert.get());
        }
        auto& to_insert = (code[node->code_length() - 1] ? current->_right : current->_left);
        to_insert.reset(node);
        return true;
    }
    void invalidate_probabilities() override
    {
        this->_probability = (this->left() ? this->left()->probability() : Probability()) + (this->right() ? this->right()->probability() : Probability());
    }

    base* take_left() { return this->_left.release(); }
    base* take_right() { return this->_right.release(); }

    static std::vector<bool> zeros(size_t length) { return std::vector<bool>(length, false); }
    static void fill_zeros(std::vector<bool>& v, size_t new_length)
    {
        if (v.size() < new_length) {
            v.insert(v.end(), new_length - v.size(), false);
        }
    }
    static void increment_binary(std::vector<bool>& v)
    {
        if (!v.empty()) {
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
    }

protected:
    huffman_tree() noexcept
        : base(Probability(), base::type::internal_node)
    {
    }

    void prepend_bit(bool bit) override
    {
        _left->prepend_bit(bit);
        _right->prepend_bit(bit);
    }

    static std::string to_string(const std::vector<bool>& v)
    {
        std::stringstream ss;
        std::copy(v.begin(), v.end(), std::ostream_iterator<bool>(ss, ""));
        return ss.str();
    }

private:
    std::unique_ptr<base> _left;
    std::unique_ptr<base> _right;
};

template <typename Data, typename Probability, typename Code = std::vector<bool>>
class huffman_node : public huffman_tree_base<Data, Probability, Code> {
    using base = huffman_tree_base<Data, Probability, Code>;
    using code_type = typename base::code_type;

public:
    huffman_node(const Data& d, const Probability& p)
        : base(p, base::type::leaf_node)
        , _data(d)
    {
    }

    Data data() const { return this->_data; }
    code_type code() const { return this->_code; }
    size_t code_length() const { return this->_code.size(); }

    void set_code(const code_type& new_code) { this->_code = new_code; }
    void set_code(code_type&& new_code) { this->_code = std::move(new_code); }

protected:
    void prepend_bit(bool bit) override
    {
        this->_code.insert(this->_code.begin(), bit);
    }

private:
    const Data _data;
    code_type _code; //TODO: keep consisten naming
};

template <typename Data, typename Probability, typename Code>
void make_canonical(huffman_tree<Data, Probability, Code>& root)
{
    if (root.is_leaf())
        return;
    using leaf_node = huffman_node<Data, Probability, Code>;
    using internal_node = huffman_tree<Data, Probability, Code>;
    // detach all leaf nodes
    std::vector<leaf_node*> nodes;
    std::deque<internal_node*> to_check;
    to_check.push_back(&root);
    while (!to_check.empty()) {
        auto parent = to_check.back();
        to_check.pop_back();
        if (parent->left()) {
            if (parent->left()->is_leaf())
                nodes.push_back(static_cast<leaf_node*>(parent->take_left()));
            else
                to_check.push_back(static_cast<internal_node*>(parent->left()));
        }
        if (parent->right()) {
            if (parent->right()->is_leaf())
                nodes.push_back(static_cast<leaf_node*>(parent->take_right()));
            else
                to_check.push_back(static_cast<internal_node*>(parent->right()));
        }
    }
    if (!nodes.empty()) {
        root.reset();
        // sort leaf nodes by code length and probability
        std::sort(nodes.begin(), nodes.end(), [](const leaf_node* left, const leaf_node* right) {
            return left->code_length() == right->code_length() ? left->probability() > right->probability() : left->code_length() < right->code_length();
        });
        // assign new canonical codes
        size_t last_length = nodes[0]->code_length();
        Code to_set = internal_node::zeros(last_length);
        for (auto& n : nodes) {
            if (n->code_length() != last_length) {
                last_length = n->code_length();
                internal_node::fill_zeros(to_set, last_length);
            }
            n->set_code(to_set);
            // insert the new node
            root.push_node(n);
            internal_node::increment_binary(to_set);
        }
        // finally rebuild the probabilites
        root.invalidate_probabilities();
    }
}

template <typename Data, typename Probability, typename Code>
template <class ForwardIt, class DataSelector, class ProbSelector>
std::unique_ptr<huffman_tree<Data, Probability, Code>> huffman_tree_base<Data, Probability, Code>::build(ForwardIt begin, ForwardIt end, const DataSelector& get_data, const ProbSelector& get_prob)
{
    if (begin == end)
        return nullptr;
    using Tree = huffman_tree_base<Data, Probability, Code>;

    struct greater_than {
        bool operator()(const huffman_tree_base* left, const huffman_tree_base* right) const
        {
            return left->_probability > right->_probability;
        }
    };

    std::priority_queue<Tree*, std::vector<Tree*>, greater_than> nodes;
    for (auto it = begin; it != end; it = std::next(it)) {
        nodes.push(new huffman_node<Data, Probability, Code>(get_data(*it), get_prob(*it)));
    }
    while (nodes.size() > 1) {
        auto left = std::unique_ptr<Tree>(nodes.top());
        nodes.pop();
        auto right = std::unique_ptr<Tree>(nodes.top());
        nodes.pop();
        nodes.push(new huffman_tree<Data, Probability, Code>(std::move(left), std::move(right)));
    }
    return std::unique_ptr<huffman_tree<Data, Probability, Code>>(static_cast<huffman_tree<Data, Probability, Code>*>(nodes.top()));
}

template <typename Data, typename Probability, typename Code = std::vector<bool>>
class huffman_encoder {
    using tree_type = huffman_tree_base<Data, Probability, Code>;

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

template <typename Data, typename Probability, typename Code>
auto make_encoder(const huffman_tree_base<Data, Probability, Code>& tree)
{
    return huffman_encoder<Data, Probability, Code>(tree);
}

} // namespace huffman
