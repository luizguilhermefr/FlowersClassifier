import tensorflow as tf

IMAGE_SIZE = 224


def process_image(numpy_image):
    img = tf.convert_to_tensor(numpy_image, dtype=tf.float32)
    resized = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    normalized = resized / 255
    return normalized.numpy()
