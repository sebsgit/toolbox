public class Test {
	static { System.loadLibrary("test"); }
	public static native long create_image(int width, int height, int stride, byte[] data, int imageOwnsData);
	public static native Frame create_frame(long index, long image);
	
	public static native int get_frame_size();
	public static native void create_frame_raw(long index, long image, byte[] result);
	public static native long get_content_raw(byte[] rawFrame);
	public static native long get_index_raw(byte[] rawFrame);
	public static native void free_frame_raw(byte[] rawFrame);
	
	public static native Frame create_frame_and_image(long index,int w, int h);
	public static native long get_content(Frame frame);
	public static native long get_index(Frame frame);
	public static native int get_width(long image);
	public static native int get_height(long image);
	public static native int get_stride(long image);
	public static native byte[] get_data(long image);
	public static native void free_frame(Frame frame);
	public static native void free_image(long image);
	public static void main(String[] args) {
		testImage();
		testFrame();
	}
	private static void expect(boolean condition, String failMessage) {
		if (!condition) {
			System.out.println(failMessage);
			System.exit(0);
		}
	}
	private static void testImage() {
		int width = 100;
		int height = 100;
		byte[] data = new byte[width * height];
		for (int y = 0 ; y<height ; ++y) {
			for (int x = 0 ; x<width ; ++x) {
				data[y*width + x] = (byte)((x+y) % 256);
			}
		}
		long image = create_image(width, height, width, data, 0);
		expect(image != 0, "image is null");
		expect(get_width(image) == width, "!width");
		expect(get_height(image) == height, "!height");
		expect(get_stride(image) == width, "!stride");
		byte[] imageData = get_data(image);
		expect(imageData.length == data.length, "!size");
		for (int i=0 ; i<imageData.length ; ++i)
			expect(imageData[i] == data[i], "!data");
		free_image(image);
		System.out.println("Image test OK.");
	}
	private static void testFrame() {
		final int frameSize = get_frame_size();
		expect(frameSize > 0, "!frame size");
		byte[] frame = new byte[frameSize];
		int width = 800;
		int height = 600;
		byte[] data = new byte[width * height];
		for (int y = 0 ; y<height ; ++y) {
			for (int x = 0 ; x<width ; ++x) {
				data[y*width + x] = (byte)((x+y) % 256);
			}
		}
		long image = create_image(width, height, width, data, 0);
		expect(image > 0, "!image");
		create_frame_raw(14, image, frame);
		expect(get_index_raw(frame) == 14, "!index");
		long content = get_content_raw(frame);
		expect(content == image, "!content");
		expect(get_width(content) == width, "!width");
		expect(get_height(content) == height, "!height");
		free_frame_raw(frame);
		System.out.println("Frame test OK.");
	}
};

class Frame {
	long index;
	long imagePtr;	
};
