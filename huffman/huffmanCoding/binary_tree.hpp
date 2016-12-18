#pragma once

#include <algorithm>
#include <cassert>
#include <memory>
#include <stack>
#include <tuple>
#include <type_traits>
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
