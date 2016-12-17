#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <cassert>
#include <type_traits>
#include <unordered_map>

namespace huffman{

template <typename T>
class binary_tree {
protected:
	template <typename NodeType>
	class dfs_iterator_base : public std::iterator<std::forward_iterator_tag, T> {
		friend class binary_tree;
		explicit dfs_iterator_base(NodeType node)
			:dfs_iterator_base(nullptr, node)
		{
		}
		dfs_iterator_base(NodeType previous, NodeType current)
			: _previous(previous)
			, _current(current)
		{
		}
	public:
		dfs_iterator_base()
			:dfs_iterator_base(nullptr, nullptr)
		{
		}
		const T operator* () const {
			return _current->value();
		}
		bool operator != (const dfs_iterator_base& other) const {
			return _current != other._current;
		}
		bool operator == (const dfs_iterator_base& other) const {
			return !(*this != other);
		}
		dfs_iterator_base& operator++ () {
			if (_current->_left) {
				_previous = _current;
				_current = _current->_left.get();
			} else if (_current->_right) {
				_previous = _current;
				_current = _current->_right.get();
			} else {
				NodeType root_node = _previous;
				NodeType result = nullptr;
				while (!result) {
					if (!root_node)
						break;
					if (root_node->_right)
						result = root_node->_right.get();
					else
						root_node = root_node->_root;
				}
				_previous = root_node;
				_current = result != _current ? result : nullptr;
			}
			return *this;
		}
		dfs_iterator_base operator++(int) const {
			dfs_iterator_base result(_previous, _current);
			this->operator++();
			return result;
		}
	protected:
		NodeType _previous;
		NodeType _current;
	};
public:
	using ptr = std::shared_ptr<binary_tree>;
	using dfs_const_iterator = dfs_iterator_base<const binary_tree*>;
	class dfs_iterator : public dfs_iterator_base<binary_tree*> {
	protected:
		friend class binary_tree;
		explicit dfs_iterator(binary_tree* node)
			:dfs_iterator(nullptr, node)
		{
		}
		dfs_iterator(binary_tree* previous, binary_tree* current)
			: dfs_iterator_base<binary_tree*>(previous, current)
		{
		}
	public:
		dfs_iterator()
			:dfs_iterator(nullptr, nullptr)
		{
		}
		T& operator *() {
			return dfs_iterator_base<binary_tree*>::_current->_data;
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

protected:
	binary_tree() {
	}
	binary_tree(T&& d)
		:_data(std::forward<T&&>(d))
	{
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
	auto result = std::find_if(tree->cbegin(), tree->cend(), [](int x) { return x == 1; });
	assert(*result == 1);
	result = std::find_if(tree->cbegin(), tree->cend(), [](int x) { return x == 12332; });
	assert(result == tree->cend());
	for (auto &it : *tree)
		it = 0;
	for (auto it = tree->cbegin() ; it != tree->cend() ; ++it)
		assert(*it == 0);
}

namespace huffman {
	template <typename T, typename Real=float>
	class probability_table {
		static_assert(std::is_floating_point<Real>::value, "probability_table entry needs to have a floating point probability.");
	public:
		virtual ~probability_table() {}
		virtual Real operator[] (const T& t) const = 0;
	};
	template <typename T, typename Real = float>
	class probability_map : public probability_table<T, Real> {
	public:
		Real operator[] (const T& t) const override {
			return _data.at(t);
		}
		Real& operator[] (const T& t) {
			return _data[t];
		}
	private:
		std::unordered_map<T, Real> _data;
	};
}

int main(int argc, char *argv[]) {
	test_tree_iterator();
	huffman::probability_map<int, float> probs;
	probs[1] = 0.14;
	probs[2] = 0.44;
	probs[3] = 0.42;
	return 0;
}
