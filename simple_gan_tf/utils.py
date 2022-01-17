import numpy as np
import glob as glob
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

def load_dataset(file_path_train, file_path_test, batch_size):
    train_images = []
    size = 64, 64
    for image_location in glob.glob(file_path_train + "/*.png"):
        image = Image.open(image_location)
        image.thumbnail(size, Image.ANTIALIAS)
        image = np.asarray(image)
        train_images.append(image)

    for image_location in glob.glob(file_path_test + "/*.png"):
        image = Image.open(image_location)
        image.thumbnail(size, Image.ANTIALIAS)
        image = np.asarray(image)
        train_images.append(image)

    train_images = np.array(train_images)
    train_images = train_images.reshape(-1, 64, 64, 1)

    train_images = (train_images - 127.5) / 127.5
    BUFFER_SIZE = train_images.shape[0]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images)).shuffle(BUFFER_SIZE).batch(batch_size)

    return train_dataset

def test_results(trained_generator):
    # I want to generate 4 different examples
    random_noise = tf.random.normal([4, noise_dim])
    pred = trained_generator(random_noise, training=False)
    fig = plt.figure(figsize=(16, 16))

    for i in range(pred.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(pred[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.show()
