#ifndef GRAPHLIB_HPP
#define GRAPHLIB_HPP

#include <set>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include <deque>
#include <stack>

namespace graphutils {

template <typename T> class Graph;

template <typename T>
class Node {
public:
	typedef std::shared_ptr<Node<T>> nodeptr_t;
	Node(const T& data = T())
		:_data(data)
	{

	}
	void setData(const T& t) {
		this->_data = t;
	}
	T& data(){
		return this->_data;
	}
	const T data() const{
		return this->_data;
	}
	void connectTo(nodeptr_t other) {
		this->_neighbours.insert(other);
	}
	void visitBfs(std::function<void(T& data)> fn) {
		std::set<Node*> visited;
		std::deque<Node*> queue;
		queue.push_back(this);
		while (!queue.empty()){
			auto n = queue.front();
			queue.pop_front();
			if (visited.find(n) == visited.end()){
				fn(n->data());
				for (auto p : n->_neighbours)
					queue.push_back(p.get());
				visited.insert(n);
			}
		}
	}
	void visitDfs(std::function<void(T& data)> fn) {
		std::set<Node*> visited;
		std::stack<Node*> stack;
		stack.push(this);
		while (!stack.empty()){
			auto n = stack.top();
			stack.pop();
			if (visited.find(n) == visited.end()){
				fn(n->data());
				for (auto p : n->_neighbours)
					stack.push(p.get());
				visited.insert(n);
			}
		}
	}
private:
	T _data;
	std::set<nodeptr_t> _neighbours;
	friend class Graph<T>;
};

template <typename T>
class Graph{
public:
	typedef typename Node<T>::nodeptr_t nodeptr_t;
	typedef Node<T> node_t;
	Graph() {

	}
	bool connectDirected(const T& data1, const T& data2) {
		auto n1 = this->node(data1);
		auto n2 = this->node(data2);
		if (n1 && n2) {
			n1->connectTo(n2);
			return true;
		}
		return false;
	}
	nodeptr_t add(const T& data) {
		auto node = nodeptr_t(new node_t(data));
		this->_nodes.insert(node);
		return node;
	}
	nodeptr_t node(const T& data) const{
		auto it = std::find_if(	this->_nodes.begin(),
								this->_nodes.end(),
								[&](const nodeptr_t& n){
									return n->data() == data;
								});
		return it != this->_nodes.end() ? *it : nullptr;
	}
	void visitBfs(std::function<void(T& data)> fn) {
		std::set<nodeptr_t> toVisit = this->_nodes;
		while (toVisit.empty() == false) {
			nodeptr_t node = *toVisit.begin();
			std::deque<nodeptr_t> queue;
			queue.push_back(node);
			while (!queue.empty()){
				auto n = queue.front();
				queue.pop_front();
				auto it = toVisit.find(n);
				if (it != toVisit.end()){
					fn(n->data());
					for (auto p : n->_neighbours)
						queue.push_back(p);
					toVisit.erase(it);
				}
			}
		}
	}
	void visitDfs(std::function<void(T& data)> fn) {
		std::set<nodeptr_t> toVisit = this->_nodes;
		while (toVisit.empty() == false) {
			nodeptr_t node = *toVisit.begin();
			std::stack<nodeptr_t> stack;
			stack.push(node);
			while (!stack.empty()){
				auto n = stack.top();
				stack.pop();
				auto it = toVisit.find(n);
				if (it != toVisit.end()){
					fn(n->data());
					for (auto p : n->_neighbours)
						stack.push(p);
					toVisit.erase(it);
				}
			}
		}
	}
private:
	std::set<nodeptr_t> _nodes;
};

template <typename T>
class GraphBuilder {
	typedef typename Graph<T>::nodeptr_t nodeptr_t;
	typedef std::vector<nodeptr_t> bucket_t;
public:
	void clear() {
		this->_graph = Graph<T>();
		this->_buckets.clear();
	}
	void add(const T& data){
		auto node = this->_graph.add(data);
		this->_buckets.push_back(bucket_t(1, node));
	}
	void connectDirected(const T& data1, const T& data2) {
		if (this->_graph.connectDirected(data1,data2) ) {
			auto n1 = this->_graph.node(data1);
			auto n2 = this->_graph.node(data2);
			int b1 = this->bucket(n1);
			int b2 = this->bucket(n2);
			if (b1 > -1 && b2 > -1 && b1 != b2) {
				_buckets[b1].push_back(n2);
				_buckets[b2].erase(std::find(_buckets[b2].begin(), _buckets[b2].end(), n2));
				if (_buckets[b2].empty()) {
					_buckets.erase(_buckets.begin() + b2);
				}
			}
		}
	}
	Graph<T> result() const{
		return this->_graph;
	}
	size_t parts() const{
		return this->_buckets.size();
	}
private:
	int bucket(const nodeptr_t& node) const{
		for (int i=0 ; i<_buckets.size() ; ++i)
			if (std::find(_buckets[i].begin(), _buckets[i].end(), node) != _buckets[i].end())
				return i;
		return -1;
	}
private:
	Graph<T> _graph;
	std::vector<bucket_t> _buckets;
};

}

#endif
