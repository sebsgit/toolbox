#include "binary_tree.hpp"
#include "huffman_encoder.hpp"
#include <iostream>
#include <iterator>

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
	auto encoded_data = enc_int.encode("abcd", 4);
	enc_int.decode(encoded_data, std::ostream_iterator<char>(std::cout, ""));
	std::cout << " = " << encoded_data.to_string() << '\n';
	std::vector<char> decoded;
	enc_int.decode(encoded_data, std::back_inserter(decoded));
	assert(decoded.size() == 4);
	assert(decoded[0] == 'a');
	assert(decoded[1] == 'b');
	assert(decoded[2] == 'c');
	assert(decoded[3] == 'd');
	assert(enc_int.decode_next(encoded_data) == 'a');
	assert(enc_int.decode_next(encoded_data) == 'b');
	assert(enc_int.decode_next(encoded_data) == 'c');
	assert(enc_int.decode_next(encoded_data) == 'd');
	return 0;
}
