import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import load_images
from model import CycleGan
from networks import generator, discriminator

file_name_root = '../input/monet2photo/'

def test(generator, test_batch):
    test_numpy = np.stack(list(test_batch))
    test_image = generator(test_numpy[0])
    plt.imshow(test_image[0, :, :, 0] * 127.5 + 127.5, cmap='gray')


def main():
    photos_test = load_images(file_name_root, 'testB')
    photos_train = load_images(file_name_root, 'trainB')
    monet_test = load_images(file_name_root, 'testA')
    monet_train = load_images(file_name_root, 'trainA')

    # Loss function for evaluating adversarial loss
    adv_loss_fn = tf.keras.losses.MeanSquaredError()

    # Define the loss function for the generators
    def generator_loss_fn(fake):
      fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
      return fake_loss


    # Define the loss function for the discriminators
    def discriminator_loss_fn(real, fake):
     real_loss = adv_loss_fn(tf.ones_like(real), real)
     fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
     return (real_loss + fake_loss) * 0.5

    generator_photos = generator()
    generator_monet = generator()
    discriminator_photos = discriminator()
    discriminator_monet = discriminator()

    model = CycleGan(generator_photos, generator_monet, discriminator_photos, discriminator_monet)

    model.compile(gen_photos_optim = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
              gen_monet_optim = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
              disc_photos_optim = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
              disc_monet_optim = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
              gen_loss = generator_loss_fn, disc_loss = discriminator_loss_fn)

    model.fit(tf.data.Dataset.zip((monet_train, photos_train)), epochs=500)

    test(generator_photos, monet_test)

if __name__ == "__main__":
    main()