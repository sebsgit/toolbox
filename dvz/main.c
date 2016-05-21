#include <stdio.h>
#include "dvz_context.h"
#include "dvz_renderer.h"

int main(int argc, char ** argv){
	(void)argc;
	(void)argv;
	dvz_context_t ctx = dvz_create_context();
	dvz_node_id_t input = dvz_register_node(ctx, "input", 0);
	dvz_node_id_t output = dvz_register_node(ctx, "output", 0);
	dvz_node_id_t modify = dvz_register_node(ctx, "modify", 0);
	dvz_node_id_t output2 = dvz_register_node(ctx, "other", 0);
	dvz_node_id_t sync = dvz_register_node(ctx, "sync", 0);
	dvz_connect(ctx, input, 0, modify, 0, DVZ_CONNECT_SOURCE);
	dvz_connect(ctx, modify, 0, output, 0, DVZ_CONNECT_SOURCE);
	dvz_connect(ctx, modify, 1, output2, 0, DVZ_CONNECT_SOURCE);
	dvz_connect(ctx, output, 0, sync, 0, DVZ_CONNECT_SOURCE);
	dvz_connect(ctx, output2, 0, sync, 1, DVZ_CONNECT_SOURCE);
	dvz_render_ascii(ctx, stdout);
	dvz_render_ppm(ctx, 400, 400, "test.ppm");
	dvz_free_context(ctx);
	return 0;
}
