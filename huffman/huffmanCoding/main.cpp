#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <cassert>

namespace huffman {

template <typename T>
class binary_tree {
public:
	using ptr = std::shared_ptr<binary_tree>;

	class dfs_const_iterator : public std::iterator<std::forward_iterator_tag, T> {
	public:
		dfs_const_iterator(const binary_tree* node = nullptr)
			:dfs_const_iterator(nullptr, node)
		{
		}
		dfs_const_iterator(const binary_tree* previous, const binary_tree* current)
			: _previous(previous)
			, _current(current)
		{
		}
		const T operator* () const {
			return _current->value();
		}
		bool operator != (const dfs_const_iterator& other) const {
			return _current != other._current;
		}
		bool operator == (const dfs_const_iterator& other) const {
			return !(*this != other);
		}
		dfs_const_iterator& operator++ () {
			if (_current->_left) {
				_previous = _current;
				_current = _current->_left.get();
			} else if (_current->_right) {
				_previous = _current;
				_current = _current->_right.get();
			} else {
				const binary_tree* root_node = _previous;
				const binary_tree* result = nullptr;
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
		dfs_const_iterator operator++(int) const {
			dfs_const_iterator result(_previous, _current);
			this->operator++();
			return result;
		}
	private:
		const binary_tree* _previous;
		const binary_tree* _current;
	};

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

	dfs_const_iterator begin() const {
		return dfs_const_iterator(this);
	}
	dfs_const_iterator end() const {
		return dfs_const_iterator(nullptr);
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
	auto result = std::find_if(tree->begin(), tree->end(), [](int x) { return x == 1; });
	assert(*result == 1);
	result = std::find_if(tree->begin(), tree->end(), [](int x) { return x == 12332; });
	assert(result == tree->end());
}

int main(int argc, char *argv[]) {
	test_tree_iterator();
	return 0;
}
