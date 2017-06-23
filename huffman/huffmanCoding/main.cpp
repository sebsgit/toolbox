#include <vector>
#include <queue>
#include <iostream>
#include <unordered_map>
#include <iterator>
#include <functional>
#include <memory>
#include <cassert>
#include <sstream>

//TODO:
// serialize / deserialize / canonical form
// configurable "code" type
// "flatten" after building - dont keep unused nodes (remove traverse_leafs)
// code / decode api
// optimality api (entropy etc.)
// debugging api (memory usage etc.)
// optimize

template <typename Data, typename Prob>
class huffman_tree;

template <typename Data, typename Prob>
class huffman_tree_base {
public:
    explicit huffman_tree_base(const Prob& p) : _prob(p) {}
    virtual ~huffman_tree_base() = default;

    virtual Prob prob() const { return this->_prob; }
    virtual Data data() const { throw std::runtime_error("data() called on non-leaf node."); }

    virtual std::vector<bool> code() const {
        throw std::runtime_error("code() called on non-leaf node.");
        return std::vector<bool>();
    }

    virtual void traverse_leafs(const std::function<void(const huffman_tree_base*)>& fn) const = 0;
    virtual void prepend_bit(bool bit) = 0;

    virtual bool isLeaf() const { return false; }

    struct greater_than
    {
        bool operator()(const huffman_tree_base* left, const huffman_tree_base* right) const {
            return left->_prob > right->_prob;
        }
    };

    template <typename ForwardIt, typename DataSelector, typename ProbSelector>
    static huffman_tree<Data, Prob>* build(ForwardIt begin, ForwardIt end, const DataSelector& get_data, const ProbSelector& get_prob);

protected:
    Prob _prob;
};

template <typename Data, typename Prob>
class huffman_tree : public huffman_tree_base<Data, Prob> {
    using base = huffman_tree_base<Data, Prob>;
public:
    huffman_tree(huffman_tree_base<Data, Prob>* left, huffman_tree_base<Data, Prob>* right)
        :base(left->prob() + right->prob())
        ,_left(left)
        ,_right(right)
    {
        _left->prepend_bit(0);
        _right->prepend_bit(1);
    }

    void traverse_leafs(const std::function<void (const huffman_tree_base<Data, Prob> *)> &fn) const override {
        if (_left)
            _left->traverse_leafs(fn);
        if (_right)
            _right->traverse_leafs(fn);
    }

    template <typename ForwardIt, typename OutputIt>
    void decode(ForwardIt begin, ForwardIt end, OutputIt out)
    {
        auto tree = this;
        for (auto it = begin; it != end; it = std::next(it)) {
            auto next = *it ? tree->_right.get() : tree->_left.get();
            if (!next)
                throw std::runtime_error("invalid code.");
            if (next->isLeaf()) {
                *out = next->data();
                ++out;
                tree = this;
            } else {
                tree = static_cast<huffman_tree*>(next);
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

template <typename Data, typename Prob>
class huffman_node : public huffman_tree_base<Data, Prob>
{
    using base = huffman_tree_base<Data, Prob>;
public:
    huffman_node(const Data& d, const Prob& p)
        : base(p)
        , _data(d)
    {

    }

    Data data() const override { return this->_data; }
    std::vector<bool> code() const override { return this->_code; }

    virtual bool isLeaf() const override { return true; }

    void traverse_leafs(const std::function<void (const base *)> &fn) const override {
        fn(this);
    }

protected:
    void prepend_bit(bool bit) override {
        this->_code.insert(this->_code.begin(), bit);
    }

private:
    const Data _data;
    std::vector<bool> _code;
};


template <typename Data, typename Prob> template <class ForwardIt, class DataSelector, class ProbSelector>
huffman_tree<Data, Prob>* huffman_tree_base<Data,Prob>::build(ForwardIt begin, ForwardIt end, const DataSelector& get_data, const ProbSelector& get_prob)
{
    if (begin == end)
        return nullptr;
    using Tree = huffman_tree_base<Data, Prob>;
    std::priority_queue<Tree*, std::vector<Tree*>, Tree::greater_than> nodes;
    for(auto it = begin ; it != end ; it = std::next(it)) {
        nodes.push(new huffman_node<Data, Prob>(get_data(*it), get_prob(*it)));
    }
    while (nodes.size() > 1) {
        Tree* left = nodes.top();
        nodes.pop();
        Tree* right = nodes.top();
        nodes.pop();
        nodes.push(new huffman_tree<Data, Prob>(left, right));
    }
    return dynamic_cast<huffman_tree<Data, Prob>*>(nodes.top());
}

static std::string code_to_str(const std::vector<bool>& code) {
    std::stringstream ss;
    for (auto b : code)
        ss << (b ? 1 : 0);
    return ss.str();
}

int main()
{
    std::unordered_map<int, float> probs;
    probs[1] = 0.10f;
    probs[2] = 0.15f;
    probs[3] = 0.30f;
    probs[4] = 0.16f;
    probs[5] = 0.29f;
    std::unordered_map<int, std::string> expected_codes;
    expected_codes[1] = "010";
    expected_codes[2] = "011";
    expected_codes[3] = "11";
    expected_codes[4] = "00";
    expected_codes[5] = "10";

    using Tree = huffman_tree_base<int ,float>;
    auto tree = std::unique_ptr<huffman_tree<int, float>>(Tree::build(probs.begin(),
                                                                      probs.end(),
                                                                      [](auto pair){ return pair.first; },
                                                                      [](auto pair){ return pair.second; })
                                                          );
    tree->traverse_leafs([&](const Tree* node) {
        assert(expected_codes[node->data()] == code_to_str(node->code()));
    });

    std::vector<bool> code{0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0};
    std::vector<int> message;
    tree->decode(code.begin(), code.end(), std::back_inserter(message));
    assert(message.size() == 5);
    assert(message == std::vector<int>({1, 2, 4, 4, 5}));
    return 0;
}
