#include "graph.hpp"
#include <iostream>
#include <cassert>

using namespace graphutils;

static void test_int(){
	Graph<int> graph;
	graph.add(1);
	auto n1 = graph.add(2);
	auto n2 = graph.add(3);
	n1->connectTo(n2);
	assert(graph.node(1));
	assert(graph.node(2));
	assert(graph.node(3));
	assert(graph.node(0) == nullptr);
	graph.visitBfs([&](int& n){
		n *= 2;
	});
	assert(graph.node(2));
	assert(graph.node(4));
	assert(graph.node(6));
	graph.visitDfs([&](int& n){
		n *= 3;
	});
	assert(graph.node(6));
	assert(graph.node(12));
	assert(graph.node(18));
}

template <typename T>
int indexOf(const std::vector<T>& v, const T& element) {
	const auto pos = std::find(v.begin(), v.end(), element);
	if (pos != v.end()){
		return std::distance(v.begin(), pos);
	}
	return -1;
}

//       1
//  2       3
//   4    5   6
static void test_visit(){
	Graph<int> graph;
	graph.add(1);
	graph.add(2);
	graph.add(3);
	graph.add(4);
	graph.add(5);
	graph.add(6);
	graph.connectDirected(1,2);
	graph.connectDirected(2,4);
	graph.connectDirected(1,3);
	graph.connectDirected(3,5);
	graph.connectDirected(3,6);
	std::vector<int> visitOrder;
	graph.visitDfs([&](int n){
		visitOrder.push_back(n);
	});
	assert(visitOrder.size() == 6);
	assert(indexOf(visitOrder, 1) == 0);
	assert(indexOf(visitOrder, 2) < indexOf(visitOrder,4));
	assert(indexOf(visitOrder, 3) < indexOf(visitOrder,6));
	assert(indexOf(visitOrder, 3) < indexOf(visitOrder,5));
	visitOrder.clear();
	graph.visitBfs([&](int n){
		visitOrder.push_back(n);
	});
	assert(visitOrder.size() == 6);
	assert(indexOf(visitOrder, 1) == 0);
	assert(indexOf(visitOrder, 2) < indexOf(visitOrder,3));
	assert(indexOf(visitOrder, 3) < indexOf(visitOrder,4));
	assert(indexOf(visitOrder, 4) < indexOf(visitOrder,5));
	assert(indexOf(visitOrder, 5) < indexOf(visitOrder,6));
	visitOrder.clear();
	graph.node(1)->visitBfs([&](int n){
		visitOrder.push_back(n);
	});
	assert(visitOrder.size() == 6);
	assert(indexOf(visitOrder, 1) == 0);
	assert(indexOf(visitOrder, 2) < indexOf(visitOrder,3));
	assert(indexOf(visitOrder, 3) < indexOf(visitOrder,4));
	assert(indexOf(visitOrder, 4) < indexOf(visitOrder,5));
	assert(indexOf(visitOrder, 5) < indexOf(visitOrder,6));
	visitOrder.clear();
	graph.node(1)->visitDfs([&](int n){
		visitOrder.push_back(n);
	});
	assert(visitOrder.size() == 6);
	assert(indexOf(visitOrder, 1) == 0);
	assert(indexOf(visitOrder, 2) < indexOf(visitOrder,4));
	assert(indexOf(visitOrder, 3) < indexOf(visitOrder,6));
	assert(indexOf(visitOrder, 3) < indexOf(visitOrder,5));
}

static void test_builder(){
	GraphBuilder<int> builder;
	builder.add(1);
	builder.add(2);
	builder.add(3);
	builder.add(4);
	builder.add(5);
	builder.add(6);
	builder.connectDirected(1,2);
	builder.connectDirected(3,4);
	builder.connectDirected(5,6);
	assert(builder.parts() == 3);
}

int main(int argc, char ** argv) {
	test_int();
	test_visit();
	test_builder();
	return 0;
}
