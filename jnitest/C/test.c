#include "Test.h"
#include "test.h"

#include <stdlib.h>

/*
 * Class:     Test
 * Method:    create_image
 * Signature: (III[Ljava/lang/Byte;I)J
 */
JNIEXPORT jlong JNICALL Java_Test_create_1image(JNIEnv * env, jclass klass, jint width, jint height, jint stride, jbyteArray byteData, jint imageOwnsData) {
	// works ok, explicit copy
	//unsigned char* data = (unsigned char*)malloc(height * stride);
	//(*env)->GetByteArrayRegion(env, byteData, 0, height * stride, data);
	
	unsigned char* data = (unsigned char*)(*env)->GetByteArrayElements(env, byteData, 0);
	jlong result = (jlong)create_image(width, height, stride, data, 0);
	(*env)->ReleaseByteArrayElements(env, byteData, data, 0);
	return result;
}

/*
 * Class:     Test
 * Method:    create_frame
 * Signature: (JJ)LFrame;
 */
JNIEXPORT jobject JNICALL Java_Test_create_1frame(JNIEnv * env, jclass klass, jlong index, jlong image) {
	return 0;
}

/*
 * Class:     Test
 * Method:    create_frame_and_image
 * Signature: (JII)LFrame;
 */
JNIEXPORT jobject JNICALL Java_Test_create_1frame_1and_1image (JNIEnv * env, jclass klass, jlong index, jint width, jint height) {
	return 0;
}

/*
 * Class:     Test
 * Method:    get_content
 * Signature: (LFrame;)J
 */
JNIEXPORT jlong JNICALL Java_Test_get_1content(JNIEnv * env, jclass klass, jobject frame) {
	return 0;
}

/*
 * Class:     Test
 * Method:    get_index
 * Signature: (LFrame;)J
 */
JNIEXPORT jlong JNICALL Java_Test_get_1index(JNIEnv * env, jclass klass, jobject frame) {
	return 0;
}

/*
 * Class:     Test
 * Method:    get_width
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_Test_get_1width(JNIEnv * env, jclass klass, jlong image) {
	return get_width((monochrome_image_t*)image);
}

/*
 * Class:     Test
 * Method:    get_height
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_Test_get_1height(JNIEnv * env, jclass klass, jlong image) {
	return get_height((monochrome_image_t*)image);
}

/*
 * Class:     Test
 * Method:    get_stride
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_Test_get_1stride(JNIEnv * env, jclass klass, jlong image) {
	return get_stride((monochrome_image_t*)image);
}

/*
 * Class:     Test
 * Method:    get_data
 * Signature: (J)[Ljava/lang/Byte;
 */
JNIEXPORT jbyteArray JNICALL Java_Test_get_1data(JNIEnv * env, jclass klass, jlong image) {
	int size = get_stride((monochrome_image_t*)image) * get_height((monochrome_image_t*)image);
	jbyteArray result = (*env)->NewByteArray(env, size);
	if (result)
		(*env)->SetByteArrayRegion(env, result, 0, size, get_data((monochrome_image_t*)image));
	return result;
}

/*
 * Class:     Test
 * Method:    free_frame
 * Signature: (LFrame;)V
 */
JNIEXPORT void JNICALL Java_Test_free_1frame(JNIEnv * env, jclass klass, jobject frame) {
	
}

/*
 * Class:     Test
 * Method:    free_image
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_Test_free_1image(JNIEnv * env, jclass klass, jlong image) {
	free_image((monochrome_image_t*)image);	
}



// `raw` methods


/*
 * Class:     Test
 * Method:    get_frame_size
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_Test_get_1frame_1size(JNIEnv * env, jclass klass) {
	return sizeof(frame_t);
}

/*
 * Class:     Test
 * Method:    create_frame_raw
 * Signature: (JJ[B)V
 */
JNIEXPORT void JNICALL Java_Test_create_1frame_1raw(JNIEnv * env, jclass klass, jlong index, jlong image, jbyteArray result) {
	frame_t temp = create_frame(index, (monochrome_image_t*)image);
	(*env)->SetByteArrayRegion(env, result, 0, sizeof(temp), (unsigned char*)&temp);
}

/*
 * Class:     Test
 * Method:    get_content_raw
 * Signature: ([B)J
 */
JNIEXPORT jlong JNICALL Java_Test_get_1content_1raw(JNIEnv * env, jclass klass, jbyteArray frame) {
	frame_t temp;
	(*env)->GetByteArrayRegion(env, frame, 0, sizeof(temp), (unsigned char*)&temp);
	return (jlong)temp.image;
}

/*
 * Class:     Test
 * Method:    get_index_raw
 * Signature: ([B)J
 */
JNIEXPORT jlong JNICALL Java_Test_get_1index_1raw(JNIEnv * env, jclass klass, jbyteArray frame) {
	frame_t temp;
	(*env)->GetByteArrayRegion(env, frame, 0, sizeof(temp), (unsigned char*)&temp);
	return temp.index;
}

/*
 * Class:     Test
 * Method:    free_frame_raw
 * Signature: ([B)V
 */
JNIEXPORT void JNICALL Java_Test_free_1frame_1raw(JNIEnv * env, jclass klass, jbyteArray frame) {
	frame_t temp;
	(*env)->GetByteArrayRegion(env, frame, 0, sizeof(temp), (unsigned char*)&temp);
	free_frame(temp);
}
