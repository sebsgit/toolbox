#ifndef _H
#define _H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	float x;
	float y;
} point2f_t;

typedef struct {
	unsigned char* data;
	int stride;
	int width;
	int height;
} monochrome_image_t;

typedef struct {
	unsigned long index;
	monochrome_image_t* image;
} frame_t;

extern monochrome_image_t* create_image(const int width, const int height, const int stride, unsigned char* data, int imageOwnsData);
extern frame_t create_frame(unsigned long index, monochrome_image_t* image);
extern frame_t create_frame_and_image(unsigned long index, const int w, const int h);
extern monochrome_image_t* get_content(frame_t frame);
extern unsigned long get_index(frame_t frame);
extern int get_width(monochrome_image_t* image);
extern int get_height(monochrome_image_t* image);
extern int get_stride(monochrome_image_t* image);
extern unsigned char* get_data(monochrome_image_t* image);
extern void free_frame(frame_t frame);
extern void free_image(monochrome_image_t* image);

#ifdef __cplusplus
}
#endif

#endif
