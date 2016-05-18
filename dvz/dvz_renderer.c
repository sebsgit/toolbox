#include "dvz_renderer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <assert.h>

/* ascii rendererer */
void dvz_render_ascii(const dvz_context_t ctx, FILE *file) {
	fprintf(file, "context has %i nodes\n", dvz_get_node_count(ctx));
	for (uint32_t i=0 ; i<dvz_get_node_count(ctx) ; ++i) {
		dvz_node_id_t node = dvz_get_node(ctx, i);
		fprintf(file, "--> %s [connections: %u]\n", dvz_get_node_name(node), dvz_get_node_connection_count(node));
		for (uint32_t n=0 ; n<dvz_get_node_connection_count(node) ; ++n) {
			const dvz_connection_id_t conn = dvz_get_node_connection(node, n);
			fprintf(file, "%s(%u) -> [%s] -> %s(%u)\n", dvz_get_node_name(dvz_get_connection_source(conn)), dvz_get_connection_source_pin(conn), dvz_get_connection_type_name(dvz_get_connection_type(conn)), dvz_get_node_name(dvz_get_connection_target(conn)), dvz_get_connection_target_pin(conn));
		}
	}
}

/* .ppm renderer */

static uint32_t min(const uint32_t a, const uint32_t b) {
	return a > b ? b : a;
}
static uint32_t max(const uint32_t a, const uint32_t b) {
	return a < b ? b : a;
}

typedef struct {
	uint8_t r, g, b;
} dvz_rgb_t;

static dvz_rgb_t dvz_rgb(const uint8_t r, const uint8_t g, const uint8_t b) {
	const dvz_rgb_t result = {r, g, b};
	return result;
}

typedef struct {
	int32_t x, y;
} dvz_point2d_t;

typedef struct {
	float x, y;
} dvz_point2df_t;

dvz_point2d_t dvz_point(const uint32_t x, const uint32_t y) {
	const dvz_point2d_t result = {x, y};
	return result;
}

typedef struct {
	dvz_rgb_t* data;
	uint32_t width, height;
} dvz_ppm3_canvas_t;

typedef struct {
	dvz_rgb_t color;
	uint32_t glyphWidth;
} dvz_pen_t;

typedef struct {
	dvz_ppm3_canvas_t* canvas;
	dvz_pen_t pen;
} dvz_ppm3_painter_t;

static void dvz_ppm3_save_canvas(dvz_ppm3_canvas_t* canvas, FILE* file) {
	assert(canvas);
	assert(file);
	fprintf(file, "P3\n%u %u\n", canvas->width, canvas->height);
	for (uint32_t y=0 ; y<canvas->height ; ++y) {
		for (uint32_t x=0 ; x<canvas->height ; ++x) {
			const dvz_rgb_t color = canvas->data[y * canvas->width + x];
			fprintf(file, "%u %u %u ", color.r, color.b, color.g);
		}
		fprintf(file, "\n");
	}
}

static dvz_ppm3_canvas_t* dvz_create_ppm_canvas(const uint32_t w, const uint32_t h) {
	assert(w && h);
	dvz_ppm3_canvas_t* result = malloc(sizeof(*result));
	result->width = w;
	result->height = h;
	result->data = malloc(w * h * sizeof(dvz_rgb_t));
	memset(result->data, 255, w * h * sizeof(dvz_rgb_t));
	return result;
}

static void dvz_free_ppm_canvas(dvz_ppm3_canvas_t* canvas) {
	assert(canvas);
	free(canvas->data);
	free(canvas);
}

static void dvz_ppm3_draw_pixel(dvz_ppm3_painter_t* painter, const uint32_t x, const uint32_t y) {
	assert(painter);
	assert(painter->canvas);
	if (x < painter->canvas->width && y < painter->canvas->height)
		painter->canvas->data[y * painter->canvas->width + x] = painter->pen.color;
}

static void dvz_ppm3_draw_line(dvz_ppm3_painter_t* painter, const uint32_t x1, const uint32_t y1,
							const uint32_t x2, const uint32_t y2)
{
	const uint32_t startX = min(x1, x2);
	const uint32_t stopX = max(x1, x2);
	const uint32_t startY = min(y1, y2);
	const uint32_t stopY = max(y1, y2);
	if (y1 == y2) {
		for (uint32_t x=startX ; x<=stopX ; ++x) {
			dvz_ppm3_draw_pixel(painter, x, y1);
		}
	} else if (x1 == x2) {
		for (uint32_t y=startY ; y<=stopY ; ++y) {
			dvz_ppm3_draw_pixel(painter, x1, y);
		}
	} else {
		const float slope = (1.0 * y2 - y1) / (1.0 * x2 - x1);
		for (uint32_t x=startX ; x<=stopX ; ++x) {
			const uint32_t y = (uint32_t)(slope * (1.0 * x - x1)) + y1;
			dvz_ppm3_draw_pixel(painter, x, y);
		}
	}
}

static void dvz_ppm3_draw_lines(dvz_ppm3_painter_t* painter, const dvz_point2d_t* points, const uint32_t count) {
	if (count > 0)
		for (uint32_t i=0 ; i<count-1 ; ++i) {
			dvz_ppm3_draw_line(painter, points[i].x, points[i].y, points[i+1].x, points[i+1].y);
		}
}

static void dvz_ppm3_draw_rect(dvz_ppm3_painter_t* painter, const uint32_t centerX, const uint32_t centerY,
							   const uint32_t width, const uint32_t height)
{
	const uint32_t left = centerX - width / 2;
	const uint32_t top = centerY - height / 2;
	dvz_ppm3_draw_line(painter, left, top, left, top + height);
	dvz_ppm3_draw_line(painter, left, top, left + width, top);
	dvz_ppm3_draw_line(painter, left + width, top, left + width, top + height);
	dvz_ppm3_draw_line(painter, left, top + height, left + width, top + height);
}

static void dvz_ppm3_fill_rect(dvz_ppm3_painter_t *painter, const uint32_t centerX, const uint32_t centerY, const uint32_t width, const uint32_t height) {
	const uint32_t left = centerX - width / 2;
	const uint32_t top = centerY - height / 2;
	for (uint32_t y = 0 ; y < height ; ++y) {
		dvz_ppm3_draw_line(painter, left, top + y, left + width, top + y);
	}
}

static void dvz_ppm3_draw_character(dvz_ppm3_painter_t* painter, const uint32_t baseX, const uint32_t baseY, const char c);

static void dvz_ppm3_draw_text(dvz_ppm3_painter_t* painter, const uint32_t baseX, const uint32_t baseY, const char* text, const uint32_t count) {
	assert(painter);
	assert(text);
	assert(baseX < painter->canvas->width);
	assert(baseY < painter->canvas->height);
	if (count > 0) {
		const uint32_t spacing = 5;
		uint32_t offset = 0;
		for (uint32_t i=0 ; i<count ; ++i) {
			dvz_ppm3_draw_character(painter, baseX + offset, baseY, text[i]);
			offset += painter->pen.glyphWidth + spacing;
		}
	}
}

void dvz_render_ppm(const dvz_context_t ctx, const uint32_t w, const uint32_t h, const char *path) {
	assert(ctx);
	assert(path);
	dvz_ppm3_canvas_t* canvas = dvz_create_ppm_canvas(w, h);
	dvz_pen_t pen = { {255, 0, 255}, 10 };
	dvz_ppm3_painter_t painter = {canvas, pen};
	dvz_ppm3_draw_pixel(&painter, 0, 0);
	dvz_ppm3_draw_line(&painter, 0, 0, w-1, h-1);
	dvz_ppm3_draw_line(&painter, w - 1, 0, 0, h-1);
	painter.pen.color = dvz_rgb(187, 32, 76);
	dvz_ppm3_draw_rect(&painter, w / 2, h / 2, w / 4, h / 4);
	painter.pen.color = dvz_rgb(17, 100, 0);
	dvz_ppm3_fill_rect(&painter, w / 2, h / 2, w / 5, h / 5);
	painter.pen.color = dvz_rgb(0, 0, 0);
	dvz_ppm3_draw_text(&painter, 10, h - 10, "A SIMPLE FONT", 13);
	FILE* file = fopen(path, "wb");
	dvz_ppm3_save_canvas(canvas, file);
	fclose(file);
	dvz_free_ppm_canvas(canvas);
}


/* simple monospace font rendering */
#define DVZ_MAX_GLYPH_NODES 9

typedef struct {
	char c;
	dvz_point2df_t pt[DVZ_MAX_GLYPH_NODES];
	uint32_t count;
} dvz_glyph_data_t;

static dvz_glyph_data_t default_font[] = {
	{'A', { {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 0.5f}, {0.0f, 0.5f} }, 6 },
	{'B', { {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.5f}, {0.5f, 0.5f}, {1.0f, 0.5f}, {1.0f, 0.0f}, {0.0f, 0.0f} }, 8 },
	{'C', { {1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f} }, 4 },
	{'D', { {0.0f, 0.0f}, {0.0f, 1.0f}, {0.5f, 1.0f}, {1.0f, 0.75f}, {1.0f, 0.25f}, {0.5f, 0.0f}, {0.0f, 0.0f} }, 7 },
	{'E', { {1.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 0.5f}, {0.5f, 0.5f}, {0.0f, 0.5f}, {0.0f, 0.0f}, {1.0f, 0.0f} }, 7 },
	{'F', { {1.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 0.5f}, {0.5f, 0.5f}, {0.0f, 0.5f}, {0.0f, 0.0f} }, 6 },
	{'G', { {1.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.5f}, {0.5f, 0.5f} }, 6 },
	{'H', { {0.0f, 1.0f}, {0.0f, 0.0f}, {0.0f, 0.5f}, {1.0f, 0.5f}, {1.0f, 1.0f}, {1.0f, 0.0f} }, 6 },
	{'I', { {0.5f, 0.0f}, {0.5f, 1.0f} }, 2 },
	{'J', { {0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.25f}, {0.5f, 0.0f}, {0.0f, 0.25f}, {0.0f, 0.5f} }, 6 },
	{'K', { {0.0f, 1.0f}, {0.0f, 0.0f}, {0.0f, 0.5f}, {1.0f, 1.0f}, {0.0f, 0.5f}, {1.0f, 0.0f} }, 6 },
	{'L', { {0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f} }, 3 },
	{'M', { {0.0f, 0.0f}, {0.0f, 1.0f}, {0.5f, 0.5f}, {1.0f, 1.0f}, {1.0f, 0.0f} }, 5 },
	{'N', { {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f} }, 4 },
	{'O', { {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f} }, 5 },
	{'P', { {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.5f}, {0.0f, 0.5f} }, 5 },
	{'Q', { {1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.0f}, {0.5f, 0.25f} }, 6 },
	{'R', { {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.5f}, {0.0f, 0.5f}, {1.0f, 0.0f} }, 6 },
	{'S', { {1.0f, 1.0f}, {0.5f, 1.0f}, {0.0f, 0.75f}, {1.0f, 0.25f}, {0.5f, 0.0f}, {0.0f, 0.0f} }, 6 },
	{'T', { {0.5f, 0.0f}, {0.5f, 1.0f}, {0.0f, 1.0f}, {1.0f, 1.0f} }, 4 },
	{'U', { {0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f} }, 4 },
	{'V', { {0.0f, 1.0f}, {0.0f, 0.5f}, {0.5f, 0.0f}, {1.0f, 0.5f}, {1.0f, 1.0f} }, 5 },
	{'W', { {0.0f, 1.0f}, {0.0f, 0.0f}, {0.5f, 0.0f}, {0.5f, 0.5f}, {0.5f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f} }, 7 },
	{'X', { {0.0f, 0.0f}, {1.0f, 1.0f}, {0.5f, 0.5f}, {0.0f, 1.0f}, {1.0f, 0.0f} }, 5 },
	{'Y', { {0.0f, 1.0f}, {0.5f, 0.5f}, {0.5f, 0.0f}, {0.5f, 0.5f}, {1.0f, 1.0f} }, 5 },
	{'Z', { {0.0f, 1.0f}, {1.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f} }, 4 }
};

static dvz_glyph_data_t dvz_find_glyph(const char c) {
	dvz_glyph_data_t result;
	result.c = ' ';
	result.count = 0;
	const uint32_t size = sizeof(default_font) / sizeof(default_font[0]);
	for (uint32_t i=0 ; i<size ; ++i) {
		if (default_font[i].c == c) {
			result = default_font[i];
			break;
		}
	}
	return result;
}

static void dvz_glyph_get_nodes(dvz_point2d_t* nodes, uint32_t* nodeCount, dvz_ppm3_painter_t* painter, const uint32_t baseX, const uint32_t baseY, const char c) {
	const uint32_t w = painter->pen.glyphWidth;
	const uint32_t h = painter->pen.glyphWidth * 2;
	const dvz_glyph_data_t glyph = dvz_find_glyph(c);
	*nodeCount = glyph.count;
	for (uint32_t i=0 ; i<glyph.count ; ++i) {
		nodes[i] = dvz_point(baseX + glyph.pt[i].x * w, baseY - glyph.pt[i].y * h);
	}
}

static void dvz_ppm3_draw_character(dvz_ppm3_painter_t *painter, const uint32_t baseX, const uint32_t baseY, const char c) {
	assert(painter);
	if (baseX >= painter->canvas->width)
		return;
	if (baseY >= painter->canvas->height)
		return;
	dvz_point2d_t points[DVZ_MAX_GLYPH_NODES];
	uint32_t count = 0;
	dvz_glyph_get_nodes(points, &count, painter, baseX, baseY, c);
	dvz_ppm3_draw_lines(painter, points, count);
}
