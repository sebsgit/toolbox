#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <sstream>
#include <stack>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace huffman {

template <typename T>
class binary_tree {
protected:
	template <typename NodeType>
	class dfs_iterator_base : public std::iterator<std::forward_iterator_tag, T> {
		friend class binary_tree;
		explicit dfs_iterator_base(NodeType current) {
			if (current)
				_stack.push(current);
		}

	public:
		dfs_iterator_base()
			: dfs_iterator_base(nullptr) {
		}
		const T operator*() const {
			return _stack.top()->value();
		}
		bool operator!=(const dfs_iterator_base& other) const {
			return !(*this == other);
		}
		bool operator==(const dfs_iterator_base& other) const {
			return this->_stack == other._stack;
		}
		dfs_iterator_base& operator++() {
			if (_stack.empty() == false) {
				auto current = _stack.top();
				_stack.pop();
				if (current->_right)
					_stack.push(current->_right.get());
				if (current->_left)
					_stack.push(current->_left.get());
			}
			return *this;
		}
		dfs_iterator_base operator++(int)const {
			dfs_iterator_base result(_stack.empty() ? nullptr : _stack.top());
			result._stack = this->_stack;
			this->operator++();
			return result;
		}
		bool is_leaf() const {
			return _stack.empty() ? false : _stack.top()->is_leaf();
		}

	protected:
		std::stack<NodeType> _stack;
	};

public:
	using ptr = std::shared_ptr<binary_tree>;
	using dfs_const_iterator = dfs_iterator_base<const binary_tree*>;
	class dfs_iterator : public dfs_iterator_base<binary_tree*> {
	protected:
		friend class binary_tree;
		explicit dfs_iterator(binary_tree* node)
			: dfs_iterator_base<binary_tree*>(node) {
		}

	public:
		dfs_iterator()
			: dfs_iterator(nullptr) {
		}
		T& operator*() {
			return dfs_iterator_base<binary_tree*>::_stack.top()->_data;
		}
	};
	using iterator = dfs_iterator;
	using const_iterator = dfs_const_iterator;

	static ptr make_node() {
		return ptr(new binary_tree);
	}
	static ptr make_node(T&& d) {
		return ptr(new binary_tree(std::forward<T&&>(d)));
	}
	void set_value(T&& d) {
		_data = d;
	}
	void emplace_value(T&& d) {
		_data = std::move(d);
	}
	T value() const {
		return _data;
	}
	void set_left(ptr p) {
		_left = p;
		_left->_root = this;
	}
	ptr push_left(T&& d) {
		auto result = make_node(std::forward<T&&>(d));
		this->set_left(result);
		return result;
	}
	void set_right(ptr p) {
		_right = p;
		_right->_root = this;
	}
	ptr push_right(T&& d) {
		auto result = make_node(std::forward<T&&>(d));
		this->set_right(result);
		return result;
	}
	const_iterator cbegin() const {
		return const_iterator(this);
	}
	const_iterator cend() const {
		return const_iterator(nullptr);
	}
	iterator begin() {
		return iterator(this);
	}
	iterator end() const {
		return iterator(nullptr);
	}
	bool is_leaf() const {
		return !_left && !_right;
	}

protected:
	binary_tree() {}
	binary_tree(T&& d)
		: _data(std::forward<T&&>(d)) {
	}

private:
	T _data;
	binary_tree* _root = nullptr;
	ptr _left;
	ptr _right;
};
}

static void test_tree_iterator() {
	auto tree = huffman::binary_tree<int>::make_node();
	tree->set_value(2);
	tree->push_left(1);
	auto right = tree->push_right(3);
	right->push_right(12);
	right->push_left(17);
	std::vector<int> tree_order;
	for (auto it : *tree)
		tree_order.push_back(it);
	assert(tree_order.size() == 5);
	assert(tree_order.at(0) == 2);
	assert(tree_order.at(1) == 1);
	assert(tree_order.at(2) == 3);
	assert(tree_order.at(3) == 17);
	assert(tree_order.at(4) == 12);
	auto result =
		std::find_if(tree->cbegin(), tree->cend(), [](int x) { return x == 1; });
	assert(*result == 1);
	result = std::find_if(tree->cbegin(), tree->cend(),
						  [](int x) { return x == 12332; });
	assert(result == tree->cend());
	for (auto& it : *tree)
		it = 0;
	for (auto it = tree->cbegin(); it != tree->cend(); ++it)
		assert(*it == 0);
}

namespace huffman {
template <typename T, typename Real = float>
class probability_table {
	static_assert(
		std::is_arithmetic<Real>::value,
		"probability_table entry needs to have an arithmetic probability.");

public:
	using value_type = T;
	using probability_type = Real;
	using const_iterator = typename std::unordered_map<T, Real>::const_iterator;
	using iterator = typename std::unordered_map<T, Real>::iterator;

	Real operator[](const T& t) const {
		return _data.at(t);
	}
	Real& operator[](const T& t) {
		return _data[t];
	}
	const_iterator cbegin() const {
		return _data.cbegin();
	}
	const_iterator cend() const {
		return _data.cend();
	}
	iterator begin() {
		return _data.begin();
	}
	iterator end() {
		return _data.end();
	}
	size_t empty() const {
		return _data.empty();
	}

private:
	std::unordered_map<T, Real> _data;
};
class code {
public:
	code& prepend(uint8_t bit) {
		_data.insert(_data.begin(), bit);
		return *this;
	}
	std::string to_string() const {
		std::stringstream result;
		for (auto i : _data)
			result << std::to_string(i);
		return result.str();
	}

private:
	std::vector<uint8_t> _data;
};
template <typename T, typename Real = float>
class encoder {
	using probability_map = probability_table<T, Real>;

public:
	void set_probability_table(const probability_map& table) {
		_probabilities = table;
		if (!_probabilities.empty())
			this->recalculate();
	}
	code get_code(const T& value) const {
		return this->_codebook.at(value);
	}

protected:
	void recalculate() {
		assert(_probabilities.empty() == false);
		using node_type = std::tuple<T, Real, code>;
		using tree = typename binary_tree<node_type>::ptr;
		auto comparator = [](const tree& left, const tree& right) {
			return std::get<1>(left->value()) < std::get<1>(right->value());
		};
		auto erase_min = [comparator](std::vector<tree>& d) {
			auto min0 = std::min_element(d.begin(), d.end(), comparator);
			auto result = *min0;
			d.erase(min0);
			return result;
		};
		auto prepend_leafs = [](tree t, uint8_t n) {
			for (auto it = t->begin(); it != t->end(); ++it)
				if (it.is_leaf()) {
					auto& code = std::get<2>(*it);
					code.prepend(n);
				}
		};
		std::vector<tree> data;
		for (auto it : _probabilities)
			data.push_back(binary_tree<node_type>::make_node(std::make_tuple(it.first, it.second, code())));
		while (data.size() > 1) {
			auto min0_value = erase_min(data);
			auto min1_value = erase_min(data);
			auto new_probability = std::get<1>(min0_value->value()) + std::get<1>(min1_value->value());
			node_type new_node = std::make_tuple(T(), Real(new_probability), code());
			auto root = binary_tree<node_type>::make_node(std::move(new_node));
			prepend_leafs(min0_value, 0);
			prepend_leafs(min1_value, 1);
			root->set_left(min0_value);
			root->set_right(min1_value);
			data.push_back(root);
		}
		this->_codebook.clear();
		for (auto it = data[0]->cbegin(); it != data[0]->cend(); ++it)
			if (it.is_leaf())
				this->_codebook.insert({std::get<0>(*it), std::get<2>(*it)});
	}

private:
	probability_map _probabilities;
	std::unordered_map<T, code> _codebook;
};
}

int main(int argc, char* argv[]) {
	test_tree_iterator();
	huffman::probability_table<int, float> probs;
	probs[1] = 0.14;
	probs[2] = 0.44;
	probs[7] = 0.23;
	probs[32] = 0.19;
	huffman::encoder<int, float> encoder;
	encoder.set_probability_table(probs);
	for (auto it : probs)
		std::cout << it.first << ' ' << encoder.get_code(it.first).to_string() << '\n';

	huffman::probability_table<char, int> probs_int;
	for (char c = 'a'; c != 'z'; ++c)
		probs_int[c] = static_cast<int>(c);
	huffman::encoder<char, int> enc_int;
	enc_int.set_probability_table(probs_int);
	for (auto it : probs_int)
		std::cout << it.first << ' ' << enc_int.get_code(it.first).to_string() << '\n';

	return 0;
}
