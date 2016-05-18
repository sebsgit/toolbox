#include "dvz_context.h"
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#ifndef DVZ_MAX_NODES
#define DVZ_MAX_NODES (uint16_t)256
#endif
#ifndef DVZ_MAX_CONNECTIONS
#define DVZ_MAX_CONNECTIONS 512
#endif

static uint32_t max(const uint32_t a, const uint32_t b) {
	return a > b ? a : b;
}

typedef struct dvz_node_impl_t_ {
	const void* data;
	const char* name;
	dvz_node_id_t id;
	uint32_t numInputPins;
	uint32_t numOutputPins;
	dvz_connection_id_t connections[DVZ_MAX_NODES];
	uint32_t usedConnectedTo;
} dvz_node_impl_t;

typedef struct {
	dvz_node_impl_t* node1;
	dvz_node_impl_t* node2;
	dvz_connection_type_t type;
	uint16_t inputPin;
	uint16_t outputPin;
} dvz_connection_impl_t;

typedef struct {
	dvz_node_impl_t nodes[DVZ_MAX_NODES];
	dvz_connection_impl_t conn[DVZ_MAX_CONNECTIONS];
	uint16_t usedNodes;
	uint16_t usedConnections;
} dvz_context_impl_t;


dvz_context_t dvz_create_context() {
	dvz_context_impl_t* ctx = malloc(sizeof(*ctx));
	memset(ctx, 0, sizeof(*ctx));
	return (dvz_context_t)ctx;
}

void dvz_free_context(dvz_context_t ctx) {
	free(ctx);
}

dvz_node_id_t dvz_register_node(dvz_context_t ctx_ptr, const char *name, const void *data) {
	dvz_context_impl_t* ctx = (dvz_context_impl_t*)ctx_ptr;
	if (ctx) {
		if (ctx->usedNodes >= DVZ_MAX_NODES)
			return (dvz_node_id_t)0;
		dvz_node_impl_t* node = (ctx->nodes + ctx->usedNodes);
		node->data = data;
		node->id = (dvz_node_id_t)node;
		node->name = name;
		++ctx->usedNodes;
		return node->id;
	}
	return (dvz_node_id_t)NULL;
}

dvz_connection_id_t dvz_connect(dvz_context_t ctx_ptr, const dvz_node_id_t node1, const uint32_t pin1, const dvz_node_id_t node2, const uint32_t pin2, const dvz_connection_type_t type) {
	dvz_context_impl_t* ctx = (dvz_context_impl_t*)ctx_ptr;
	if (ctx) {
		if (ctx->usedConnections >= DVZ_MAX_CONNECTIONS || type == DVZ_NOT_CONNECTED)
			return (dvz_connection_id_t)0;
		dvz_connection_impl_t* conn = (ctx->conn + ctx->usedConnections);
		dvz_node_impl_t* node1_impl = (dvz_node_impl_t*)node1;
		dvz_node_impl_t* node2_impl = (dvz_node_impl_t*)node2;
		conn->type = type;
		++ctx->usedConnections;
		if (type & DVZ_CONNECT_SOURCE) {
			node1_impl->numOutputPins = max(node1_impl->numOutputPins + 1, pin1);
			node2_impl->numInputPins = max(node2_impl->numInputPins + 1, pin2);
			conn->inputPin = pin1;
			conn->outputPin = pin2;
			conn->node1 = node1;
			conn->node2 = node2;
		}
		if (type & DVZ_CONNECT_TARGET) {
			node1_impl->numInputPins = max(node1_impl->numInputPins + 1, pin1);
			node2_impl->numOutputPins = max(node2_impl->numOutputPins + 1, pin2);
			conn->inputPin = pin2;
			conn->outputPin = pin1;
			conn->node1 = node2;
			conn->node2 = node1;
		}
		node1_impl->connections[node1_impl->usedConnectedTo] = conn;
		++node1_impl->usedConnectedTo;
		node2_impl->connections[node2_impl->usedConnectedTo] = conn;
		++node2_impl->usedConnectedTo;
		return conn;
	}
	return (dvz_connection_id_t)NULL;
}

uint32_t dvz_get_node_count(const dvz_context_t ctx) {
	return ((dvz_context_impl_t*)ctx)->usedNodes;
}

dvz_node_id_t dvz_get_node(const dvz_context_t ctx_ptr, const uint32_t n) {
	dvz_context_impl_t* ctx = (dvz_context_impl_t*)ctx_ptr;
	if (ctx->usedNodes <= n)
		return (dvz_node_id_t)NULL;
	return ctx->nodes[n].id;
}

const char* dvz_get_node_name(const dvz_node_id_t id) {
	const dvz_node_impl_t* node = (const dvz_node_impl_t*)id;
	return node ? node->name : (const char*)NULL;
}

uint32_t dvz_get_node_connection_count(const dvz_node_id_t node_ptr) {
	const dvz_node_impl_t* node = (const dvz_node_impl_t*)node_ptr;
	return node ? node->usedConnectedTo : 0;
}

uint32_t dvz_get_node_output_pin_count(const dvz_node_id_t node_ptr) {
	const dvz_node_impl_t* node = (const dvz_node_impl_t*)node_ptr;
	assert(node);
	return node->numOutputPins;
}

uint32_t dvz_get_node_input_pin_count(const dvz_node_id_t node_ptr) {
	const dvz_node_impl_t* node = (const dvz_node_impl_t*)node_ptr;
	assert(node);
	return node->numInputPins;
}

dvz_connection_id_t dvz_get_node_connection(const dvz_node_id_t node_ptr, const uint32_t n) {
	const dvz_node_impl_t* node = (const dvz_node_impl_t*)node_ptr;
	if (node && n < node->usedConnectedTo)
		return node->connections[n];
	return (dvz_connection_id_t)NULL;
}

dvz_connection_type_t dvz_get_connection_type(const dvz_connection_id_t id) {
	return id ? ((dvz_connection_impl_t*)id)->type : DVZ_NOT_CONNECTED;
}

const char* dvz_get_connection_type_name(const dvz_connection_type_t type) {
	if (type == DVZ_NOT_CONNECTED)
		return "DVZ_NOT_CONNECTED";
	else if (type == DVZ_CONNECT_SOURCE)
		return "DVZ_CONNECT_SOURCE";
	else if (type == DVZ_CONNECT_BIDIRECTIONAL)
		return "DVZ_CONNECT_BIDIRECTIONAL";
	else if (type == DVZ_CONNECT_TARGET)
		return "DVZ_CONNECT_TARGET";
	return "";
}

dvz_node_id_t dvz_get_connection_source(const dvz_connection_id_t id) {
	return id ? ((dvz_connection_impl_t*)id)->node1 : (dvz_node_id_t)NULL;
}

dvz_node_id_t dvz_get_connection_target(const dvz_connection_id_t id) {
	return id ? ((dvz_connection_impl_t*)id)->node2 : (dvz_node_id_t)NULL;
}

uint32_t dvz_get_connection_source_pin(const dvz_connection_id_t id) {
	return id ? ((dvz_connection_impl_t*)id)->inputPin : 0;
}

uint32_t dvz_get_connection_target_pin(const dvz_connection_id_t id) {
	return id ? ((dvz_connection_impl_t*)id)->outputPin : 0;
}
