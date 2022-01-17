import tensorflow as tf

class CycleGan(tf.keras.Model):
    def __init__(self, generator_photos, generator_monet, discriminator_photos, discriminator_monet, lambda_cycle=10.0):
        super(CycleGan, self).__init__()
        self.gen_photos = generator_photos
        self.gen_monet = generator_monet
        self.disc_photos = discriminator_photos
        self.disc_monet = discriminator_monet
        self.lambda_cycle = lambda_cycle

    def compile(self, gen_photos_optim, gen_monet_optim, disc_photos_optim, disc_monet_optim, gen_loss, disc_loss):
        super(CycleGan, self).compile()
        self.gen_photos_optim = gen_photos_optim
        self.gen_monet_optim = gen_monet_optim
        self.disc_photos_optim = disc_photos_optim
        self.disc_monet_optim = disc_monet_optim
        self.gen_loss = gen_loss
        self.disc_loss = disc_loss
        self.cycle_loss = tf.keras.losses.MeanAbsoluteError()

    def train_step(self, batch_data):
        real_monet, real_photos = batch_data
        with tf.GradientTape(persistent=True) as tape:
            # first we pass real images to the generator to get fake images
            fake_photos = self.gen_photos(real_monet, training=True)
            fake_monet = self.gen_monet(real_photos, training=True)

            # then we create cycled images by passing our fake images to the generator again
            cycled_monet = self.gen_photos(fake_photos, training=True)
            cycled_photos = self.gen_monet(fake_monet, training=True)

            # the discriminator

            disc_real_monet = self.disc_monet(real_monet, training=True)
            disc_fake_monet = self.disc_monet(fake_monet, training=True)

            disc_real_photos = self.disc_photos(real_photos, training=True)
            disc_fake_photos = self.disc_photos(fake_photos, training=True)

            # the generator loss

            gen_loss_monet = self.gen_loss(disc_fake_monet)
            gen_loss_photos = self.gen_loss(disc_fake_photos)

            # the cycled loss(is also the part of the generator loss)
            cycled_loss_monet = self.cycle_loss(real_monet, cycled_monet) * self.lambda_cycle
            cycled_loss_photos = self.cycle_loss(real_photos, cycled_photos) * self.lambda_cycle

            total_gen_loss_monet = gen_loss_monet + cycled_loss_monet
            total_gen_loss_photos = gen_loss_photos + cycled_loss_photos

            # the discriminator loss

            disc_monet_loss = self.disc_loss(disc_fake_monet, disc_real_monet)
            disc_photos_loss = self.disc_loss(disc_fake_photos, disc_real_photos)

        # get the gradients for the generator and the discriminator
        grad_gen_monet = tape.gradient(total_gen_loss_monet, self.gen_monet.trainable_variables)
        grad_gen_photos = tape.gradient(total_gen_loss_photos, self.gen_photos.trainable_variables)

        grad_disc_monet = tape.gradient(disc_monet_loss, self.disc_monet.trainable_variables)
        grad_disc_photos = tape.gradient(disc_photos_loss, self.disc_photos.trainable_variables)

        # update the weights of the generators and the discriminators

        self.gen_photos_optim.apply_gradients(zip(grad_gen_photos, self.gen_photos.trainable_variables))
        self.gen_monet_optim.apply_gradients(zip(grad_gen_monet, self.gen_monet.trainable_variables))

        self.disc_monet_optim.apply_gradients(zip(grad_disc_monet, self.disc_photos.trainable_variables))
        self.disc_photos_optim.apply_gradients(zip(grad_disc_photos, self.disc_photos.trainable_variables))

        return {'gen_loss_photos': total_gen_loss_photos, 'disc_loss_photos': disc_photos_loss,
                'gen_loss_monet': total_gen_loss_monet, 'disc_loss_monet': disc_monet_loss}
