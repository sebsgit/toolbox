#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test.h"

monochrome_image_t* create_image(const int width, const int height, const int stride, unsigned char* data, int imageOwnsData) {
	monochrome_image_t* result = (monochrome_image_t*)malloc(sizeof(monochrome_image_t));
	result->width = width;
	result->height = height;
	result->stride = stride;
	if (imageOwnsData)
		result->data = data;
	else {
		result->data = (unsigned char*)malloc(stride * height);
		memcpy(result->data, data, stride * height);
	}
	return result;
}

frame_t create_frame(unsigned long index, monochrome_image_t* image) {
	frame_t result = { index, image };
	return result;
}

frame_t create_frame_and_image(unsigned long index, const int w, const int h) {
	unsigned char* image_data = (unsigned char*)malloc(w * h);
	int i=0;
	for ( ; i<w * h ; ++i)
		image_data[i] = (13 * i + (1243 & i)) % 256;
	return create_frame(index, create_image(w, h, w, image_data, 1));
}

monochrome_image_t* get_content(frame_t frame){
	return frame.image;
}

unsigned long get_index(frame_t frame) {
	return frame.index;
}

int get_width(monochrome_image_t* image) {
	return image->width;
}

int get_height(monochrome_image_t* image) {
	return image->height;
}

int get_stride(monochrome_image_t* image) {
	return image->stride;
}

unsigned char* get_data(monochrome_image_t* image) {
	return image->data;
}

void free_frame(frame_t frame) {
	free_image(frame.image);
}

void free_image(monochrome_image_t* image) {
	free(image->data);
}
