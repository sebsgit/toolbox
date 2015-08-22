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
private:
	static bool defaultStopCondition(T&){return false;}
public:
	typedef std::shared_ptr<Node<T>> nodeptr_t;
	typedef std::function<void(T& data)> node_visit_function_t;
	typedef std::function<bool(T&)> node_visit_stop_condition_t;
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
	/* perform a BFS search starting at this node
		search condition can be used to stop the algorithm
		if stopCondition() returns 'true' then 'fn' is not called
		for the visited node and the search is stopped
	 */
	void visitBfs(	node_visit_function_t fn,
					node_visit_stop_condition_t stopCondition = defaultStopCondition)
	{
		this->do_bfs([&](Node* node){ fn(node->data()); }, stopCondition);
	}
	void visitDfs(	node_visit_function_t fn,
					node_visit_stop_condition_t stopCondition = defaultStopCondition)
	{
		this->do_dfs([&](Node* node){fn(node->data());}, stopCondition);
	}
private:
	void do_bfs(std::function<void(Node*)> fn,
				node_visit_stop_condition_t stopCondition = defaultStopCondition)
	{
		std::set<Node*> visited;
		std::deque<Node*> queue;
		queue.push_back(this);
		while (!queue.empty()){
			auto n = queue.front();
			queue.pop_front();
			if (visited.find(n) == visited.end()){
				if (stopCondition(n->data()))
					break;
				fn(n);
				for (auto p : n->_neighbours)
					queue.push_back(p.get());
				visited.insert(n);
			}
		}
	}
	void do_dfs(std::function<void(Node*)> fn,
				node_visit_stop_condition_t stopCondition = defaultStopCondition)
	{
		std::set<Node*> visited;
		std::stack<Node*> stack;
		stack.push(this);
		while (!stack.empty()){
			auto n = stack.top();
			stack.pop();
			if (visited.find(n) == visited.end()){
				if (stopCondition(n->data()))
					break;
				fn(n);
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
	void clear(){
		this->_nodes.clear();
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
	std::vector<T> shortestPath(const T& data1, const T& data2) {
		typedef std::pair<nodeptr_t, std::vector<T>> bfs_data_t;
		std::vector<T> result;
		auto n1 = this->node(data1);
		auto n2 = this->node(data2);
		if (!n1 || !n2)
			return result;
		std::set<nodeptr_t> visited;
		std::deque<bfs_data_t> queue;
		queue.push_back(std::make_pair(n1,std::vector<T>()));
		while (!queue.empty()) {
			auto n = queue.front();
			queue.pop_front();
			auto it = visited.find(n.first);
			if (n.first->data() == data2) {
				result.insert(result.end(),n.second.begin(), n.second.end());
				result.push_back(data2);
				break;
			}
			if (it == visited.end()) {
				visited.insert(n.first);
				for (auto p : n.first->_neighbours) {
					auto vec = n.second;
					vec.push_back(n.first->data());
					queue.push_back(std::make_pair(p,vec));
				}
			}
		}
		return result;
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
				for (const auto & n : _buckets[b2])
					_buckets[b1].push_back(n);
				_buckets.erase(_buckets.begin() + b2);
			}
		}
	}
	void connect(const T& data1, const T& data2) {
		this->connectDirected(data1,data2);
		this->_graph.connectDirected(data2,data1);
	}
	Graph<T> result() const {
		return this->_graph;
	}
	size_t parts() const {
		return this->_buckets.size();
	}
private:
	int bucket(const nodeptr_t& node) const {
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
