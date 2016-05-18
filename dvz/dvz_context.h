#ifndef DVZCONTEXT_H
#define DVZCONTEXT_H

#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dvz_context_t_* dvz_context_t;
typedef void* dvz_node_id_t;
typedef void* dvz_connection_id_t;

typedef enum {
	DVZ_NOT_CONNECTED = 0x00,
	DVZ_CONNECT_SOURCE = 0x01,
	DVZ_CONNECT_TARGET = 0x10,
	DVZ_CONNECT_BIDIRECTIONAL = 0x11
} dvz_connection_type_t;

extern dvz_context_t dvz_create_context();
extern void dvz_free_context(dvz_context_t ctx);
extern dvz_node_id_t dvz_register_node(dvz_context_t ctx, const char* name, const void* data);
extern dvz_connection_id_t dvz_connect(dvz_context_t ctx, const dvz_node_id_t node1, const uint32_t pin1, const dvz_node_id_t node2, const uint32_t pin2, const dvz_connection_type_t type);
extern uint32_t dvz_get_node_count(const dvz_context_t ctx);
extern dvz_node_id_t dvz_get_node(const dvz_context_t ctx, const uint32_t n);
extern const char* dvz_get_node_name(const dvz_node_id_t);
extern const void* dvz_get_node_data(const dvz_node_id_t);
extern uint32_t dvz_get_node_connection_count(const dvz_node_id_t);
extern dvz_connection_id_t dvz_get_node_connection(const dvz_node_id_t, const uint32_t n);
extern uint32_t dvz_get_node_output_pin_count(const dvz_node_id_t);
extern uint32_t dvz_get_node_input_pin_count(const dvz_node_id_t);
extern dvz_connection_type_t dvz_get_connection_type(const dvz_connection_id_t id);
extern dvz_node_id_t dvz_get_connection_source(const dvz_connection_id_t id);
extern dvz_node_id_t dvz_get_connection_target(const dvz_connection_id_t id);
extern uint32_t dvz_get_connection_source_pin(const dvz_connection_id_t id);
extern uint32_t dvz_get_connection_target_pin(const dvz_connection_id_t id);
extern const char* dvz_get_connection_type_name(const dvz_connection_type_t);

#ifdef __cplusplus
}
#endif

#endif
