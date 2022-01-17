import numpy as np
import tensorflow as tf
import glob as glob
from PIL import Image
from tqdm import tqdm
from networks import make_generator, make_discriminator
from utils import load_dataset, test_results



def main():
    file_path_train = '../input/fingers/train'
    file_path_test = '../input/fingers/test'
    BATCH_SIZE = 128


    generator = make_generator()
    discriminator = make_discriminator()

    train_dataset = load_dataset(file_path_train, file_path_test, BATCH_SIZE)

    BCE_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def generator_loss(fake_prob):
        return BCE_loss(tf.ones_like(fake_prob), fake_prob)

    def discriminator_loss(real_prob, fake_prob):
        real_loss = BCE_loss(tf.ones_like(real_prob), real_prob)
        fake_loss = BCE_loss(tf.zeros_like(fake_prob), fake_prob)
        return (real_loss + fake_loss)/2

    # 1e-4 is from the paper
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    noise_dim = 200
    num_examples_to_generate = 4

    epochs = 500
    for _ in tqdm(range(epochs)):
        for real_images in train_dataset:
            random_noise = tf.random.normal([BATCH_SIZE, noise_dim])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(random_noise, training=True)

                real_output = discriminator(real_images, training=True)
                fake_output = discriminator(generated_images, training=True)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

                gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

                generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    test_results()

if __name__ == "__main__":
    main()