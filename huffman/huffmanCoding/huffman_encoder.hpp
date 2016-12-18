#pragma once

#include "binary_tree.hpp"

#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <vector>

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
