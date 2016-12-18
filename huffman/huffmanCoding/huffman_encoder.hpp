#pragma once

#include "binary_tree.hpp"

#include <functional>
#include <sstream>
#include <type_traits>
#include <unordered_map>

namespace huffman {
class bitstream {
public:
	bitstream& operator<<(uint8_t bit) {
		_data.push_back(bit ? true : false);
		return *this;
	}
	bitstream& operator<<(const bitstream& other) {
		for (auto b : other._data)
			_data.push_back(b);
		return *this;
	}
	bitstream& append(uint8_t bit) {
		return this->operator<<(bit);
	}
	bitstream& prepend(uint8_t bit) {
		_data.insert(_data.begin(), bit ? true : false);
		return *this;
	}
	bitstream prefix(size_t count) const {
		bitstream result;
		int i = 0;
		while (count--)
			result << _data[i++];
		return result;
	}
	void remove_front(size_t count) {
		if (_data.size() >= count)
			_data.erase(_data.begin(), _data.begin() + count);
	}
	uint8_t operator[](size_t index) const {
		return _data[index];
	}
	std::string to_string() const {
		std::stringstream ss;
		for (auto b : _data)
			ss << (b ? '1' : '0');
		return ss.str();
	}
	bool operator==(const bitstream& other) const {
		return _data == other._data;
	}
	size_t hash() const {
		return std::hash<std::vector<bool>>()(_data);
	}
	size_t size() const {
		return _data.size();
	}

private:
	std::vector<bool> _data;
};
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
		_data.prepend(bit);
		return *this;
	}
	std::string to_string() const {
		return _data.to_string();
	}
	bitstream data() const {
		return _data;
	}
	size_t size() const {
		return _data.size();
	}
	bool operator==(const code& other) const {
		return _data == other._data;
	}

private:
	bitstream _data;
};

class code_hash {
public:
	size_t operator()(const code& c) const {
		return c.data().hash();
	}
};


template <typename T, typename Real = float>
class encoder {
	using probability_map = probability_table<T, Real>;

public:
	encoder() {
	}

	void set_probability_table(const probability_map& table) {
		_probabilities = table;
		if (!_probabilities.empty())
			this->recalculate();
	}
	code get_code(const T& value) const {
		return this->_codebook.at(value);
	}
	bitstream encode(const T* data, size_t size) const {
		bitstream result;
		for (size_t i = 0; i < size; ++i)
			result << get_code(data[i]).data();
		return result;
	}
	template <typename OutputIt>
	void decode(const bitstream& stream, OutputIt output) const {
		bitstream tmp = stream;
		while (tmp.size() > 0) {
			*output = this->decode_next(tmp);
			++output;
		}
	}
	T decode_next(bitstream& stream) const {
		bitstream pref = stream.prefix(this->_min_code_length);
		auto search = [&pref](const auto& p) { return p.first.data() == pref; };
		auto it = std::find_if(_reverse_codebook.begin(), _reverse_codebook.end(), search);
		while ((it == _reverse_codebook.end()) && pref.size() < stream.size()) {
			auto next_index = pref.size();
			pref << stream[next_index];
			it = std::find_if(_reverse_codebook.begin(), _reverse_codebook.end(), search);
		}
		if (it != _reverse_codebook.end()) {
			stream.remove_front(pref.size());
			return it->second;
		} else {
			throw std::invalid_argument("can't find entry for code " + pref.to_string());
		}
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
		this->_reverse_codebook.clear();
		this->_min_code_length = std::numeric_limits<size_t>::max();
		for (auto it = data[0]->bfs_cbegin(); it != data[0]->bfs_cend(); ++it)
			if (it.is_leaf()) {
				const auto value = std::get<0>(*it);
				const auto code = std::get<2>(*it);
				this->_codebook.insert({value, code});
				this->_reverse_codebook.insert({code, value});
				if (code.size() < _min_code_length)
					_min_code_length = code.size();
			}
	}

private:
	probability_map _probabilities;
	std::unordered_map<T, code> _codebook;
	std::unordered_map<code, T, code_hash> _reverse_codebook;
	size_t _min_code_length = 0;
};
}
