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
	template <typename NodeType, template <typename...> class Container>
	class iterator_backend_adapter;
	template <typename NodeType>
	class iterator_backend_adapter<NodeType, std::stack> {
	public:
		bool empty() const {
			return _stack.empty();
		}
		NodeType first() const {
			return _stack.top();
		}
		void push(NodeType n) {
			_stack.push(n);
		}
		void split() {
			assert(_stack.empty() == false);
			auto current = _stack.top();
			_stack.pop();
			if (current->_right)
				this->push(current->_right.get());
			if (current->_left)
				this->push(current->_left.get());
		}
		bool operator==(const iterator_backend_adapter& other) const {
			return _stack == other._stack;
		}

	private:
		std::stack<NodeType> _stack;
	};
	template <typename NodeType>
	class iterator_backend_adapter<NodeType, std::vector> {
	public:
		bool empty() const {
			return _queue.empty();
		}
		NodeType first() const {
			return _queue[0];
		}
		void push(NodeType n) {
			_queue.push_back(n);
		}
		void split() {
			assert(_queue.empty() == false);
			auto current = this->_queue[0];
			_queue.erase(_queue.begin());
			if (current->_left)
				this->push(current->_left.get());
			if (current->_right)
				this->push(current->_right.get());
		}
		bool operator==(const iterator_backend_adapter& other) const {
			return _queue == other._queue;
		}

	private:
		std::vector<NodeType> _queue;
	};

	template <typename NodeType, template <typename...> class Container>
	class read_only_iterator : public std::iterator<std::forward_iterator_tag, T> {
		friend class binary_tree;
		using queue = iterator_backend_adapter<NodeType, Container>;
		explicit read_only_iterator(NodeType current) {
			if (current)
				_visit_order.push(current);
		}
		explicit read_only_iterator(const queue& visit_queue)
			: _visit_order(visit_queue) {
		}

	public:
		const T operator*() const {
			return _visit_order.first()->value();
		}
		bool operator!=(const read_only_iterator& other) const {
			return !(*this == other);
		}
		bool operator==(const read_only_iterator& other) const {
			return this->_visit_order == other._visit_order;
		}
		read_only_iterator& operator++() {
			if (_visit_order.empty() == false) {
				_visit_order.split();
			}
			return *this;
		}
		read_only_iterator operator++(int)const {
			read_only_iterator result(this->_visit_order);
			this->operator++();
			return result;
		}
		bool is_leaf() const {
			return _visit_order.empty() ? false : _visit_order.first()->is_leaf();
		}

	protected:
		queue _visit_order;
	};
	template <typename NodeType, template <typename...> class Container>
	class read_write_iterator : public read_only_iterator<NodeType, Container> {
		friend class binary_tree;
		explicit read_write_iterator(NodeType current)
			: read_only_iterator<NodeType, Container>(current) {
		}

	public:
		T& operator*() {
			return read_only_iterator<NodeType, Container>::_visit_order.first()->_data;
		}
	};

public:
	using ptr = std::shared_ptr<binary_tree>;
	using dfs_const_iterator = read_only_iterator<const binary_tree*, std::stack>;
	using bfs_const_iterator = read_only_iterator<const binary_tree*, std::vector>;
	using dfs_iterator = read_write_iterator<binary_tree*, std::stack>;
	using bfs_iterator = read_write_iterator<binary_tree*, std::vector>;

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
		return dfs_cbegin();
	}
	const_iterator cend() const {
		return dfs_cend();
	}
	iterator begin() {
		return dfs_begin();
	}
	iterator end() const {
		return dfs_end();
	}
	bfs_iterator bfs_begin() {
		return bfs_iterator(this);
	}
	bfs_iterator bfs_end() const {
		return bfs_iterator(nullptr);
	}
	dfs_iterator dfs_begin() {
		return dfs_iterator(this);
	}
	dfs_iterator dfs_end() const {
		return dfs_iterator(nullptr);
	}
	bfs_const_iterator bfs_cbegin() const {
		return bfs_const_iterator(this);
	}
	bfs_const_iterator bfs_cend() const {
		return bfs_const_iterator(nullptr);
	}
	dfs_const_iterator dfs_cbegin() const {
		return dfs_const_iterator(this);
	}
	dfs_const_iterator dfs_cend() const {
		return dfs_const_iterator(nullptr);
	}

	bool is_leaf() const {
		return !_left && !_right;
	}

protected:
	binary_tree() {}
	binary_tree(T&& d)
		: _data(std::move(d)) {
	}
	binary_tree(const T& d)
		: _data(d) {
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

	tree = huffman::binary_tree<int>::make_node(1);
	right = huffman::binary_tree<int>::make_node(2);
	right->push_left(4)->push_right(8);
	tree->push_left(7)->push_left(3)->push_left(3)->push_left(1)->push_right(9);
	tree->set_right(right);
	tree_order.clear();
	for (auto it = tree->bfs_cbegin(); it != tree->bfs_cend(); ++it)
		tree_order.push_back(*it);
	assert(tree_order.size() == 9);
	assert(tree_order.at(0) == 1);
	assert(tree_order.at(1) == 7);
	assert(tree_order.at(2) == 2);
	assert(tree_order.at(3) == 3);
	assert(tree_order.at(4) == 4);
	assert(tree_order.at(5) == 3);
	assert(tree_order.at(6) == 8);
	assert(tree_order.at(7) == 1);
	assert(tree_order.at(8) == 9);
	tree_order.clear();
	for (auto it = tree->dfs_cbegin(); it != tree->dfs_cend(); ++it)
		tree_order.push_back(*it);
	assert(tree_order.size() == 9);
	assert(tree_order.at(0) == 1);
	assert(tree_order.at(1) == 7);
	assert(tree_order.at(2) == 3);
	assert(tree_order.at(3) == 3);
	assert(tree_order.at(4) == 1);
	assert(tree_order.at(5) == 9);
	assert(tree_order.at(6) == 2);
	assert(tree_order.at(7) == 4);
	assert(tree_order.at(8) == 8);

	auto result = std::find_if(tree->cbegin(), tree->cend(), [](int x) { return x == 1; });
	assert(*result == 1);
	result = std::find_if(tree->cbegin(), tree->cend(), [](int x) { return x == 12332; });
	assert(result == tree->cend());
	for (auto& it : *tree)
		it = 0;
	for (auto it = tree->cbegin(); it != tree->cend(); ++it)
		assert(*it == 0);
}

namespace huffman {
template <typename T, typename Real = float>
class probability_table {
	static_assert(std::is_arithmetic<Real>::value,
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
		for (auto it = data[0]->bfs_cbegin(); it != data[0]->bfs_cend(); ++it)
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
