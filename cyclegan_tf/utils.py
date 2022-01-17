import glob as glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm


def load_images(file_name_root, image_type):
    images_list = []
    for image_location in tqdm(glob.glob(file_name_root + image_type +"/*.jpg")):
        image = load_img(image_location, target_size = (size, size))
        image = img_to_array(image)
        image = (image / 127.5) - 1
        images_list.append(image)
    images_list = tf.data.Dataset.from_tensor_slices((images_list)).batch(BATCH_SIZE)
    return images_list

class ReflectionPadding2D(tf.keras.layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")