#include <stdio.h>
#include "dvz_context.h"
#include "dvz_renderer.h"

int main(int argc, char ** argv){
	(void)argc;
	(void)argv;
	dvz_context_t ctx = dvz_create_context();
	dvz_node_id_t node1 = dvz_register_node(ctx, "input", 0);
	dvz_node_id_t node2 = dvz_register_node(ctx, "output", 0);
	dvz_node_id_t node3 = dvz_register_node(ctx, "modify", 0);
	dvz_node_id_t node4 = dvz_register_node(ctx, "output two", 0);
	dvz_connect(ctx, node1, 0, node3, 1, DVZ_CONNECT_SOURCE);
	dvz_connect(ctx, node3, 1, node2, 0, DVZ_CONNECT_SOURCE);
	dvz_connect(ctx, node3, 0, node4, 0, DVZ_CONNECT_SOURCE);
	dvz_render_ascii(ctx, stdout);
	dvz_render_ppm(ctx, 400, 400, "test.ppm");
	dvz_free_context(ctx);
	return 0;
}
