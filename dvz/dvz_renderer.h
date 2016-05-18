#ifndef DVZRENDERER_H
#define DVZRENDERER_H

#include "dvz_context.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void dvz_render_ascii(const dvz_context_t ctx, FILE* output);
extern void dvz_render_ppm(const dvz_context_t ctx, const uint32_t width, const uint32_t height, const char* path);

#ifdef __cplusplus
}
#endif

#endif
