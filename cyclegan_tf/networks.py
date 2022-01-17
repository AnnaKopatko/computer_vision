from utils import ReflectionPadding2D
import tensorflow as tf
import tensorflow_addons as tfa


#THE DISCRIMINATOR

activation_function = tf.keras.layers.LeakyReLU(0.2)

def conv_block(x, filters, normed, activation):
    x = tf.keras.layers.Conv2D(
        filters,
        (4, 4),
        (2, 2),
        kernel_initializer=kernel_init,
        padding='same',
        use_bias=False,
    )(x)
    if activation:
        x = activation_function(x)
    if normed:
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    return x

# Weights initializer for the layers.
kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

def discriminator(models_shape):
    input_data = tf.keras.layers.Input(shape = models_shape)
    x =  conv_block(input_data, 64, normed = False, activation = False)
    x =  conv_block(x, 128, normed = True, activation = True)
    x =  conv_block(x, 256, normed = True, activation = True)
    x =  conv_block(x, 512, normed = True, activation = True)
    output_data = tf.keras.layers.Conv2D(1, (4, 4), (1, 1), padding = 'same', kernel_initializer=kernel_init
    )(x)
    model = tf.keras.models.Model(input_data, output_data)
    #print(model.summary())
    return model

discriminator_photos = discriminator()
discriminator_monet = discriminator()

#THE GENERATOR

def downsample(x, filters):
    x = tf.keras.layers.Conv2D(
        filters,
        (3, 3),
        (2, 2),
        kernel_initializer=kernel_init,
        padding='same',
        use_bias=False,
    )(x)
    x = activation_function(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    return x




def upsample(x, filters):
    x = tf.keras.layers.Conv2DTranspose(filters, (3,3), (2, 2), padding = 'same', kernel_initializer =kernel_init)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    x = activation_function(x)
    return x


def residual_block(x):
    # the shape doesn't change
    new_shape = x.shape[-1]

    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = tf.keras.layers.Conv2D(new_shape, (3, 3), (1, 1), kernel_initializer=kernel_init)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    x = activation_function(x)

    x = ReflectionPadding2D()(x)
    x = tf.keras.layers.Conv2D(new_shape, (3, 3), (1, 1), kernel_initializer=kernel_init)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)

    x = tf.keras.layers.add([input_tensor, x])
    return x


def generator(models_shape):
    input_data = tf.keras.layers.Input(shape=models_shape)
    x = tf.keras.layers.Conv2D(64, (7, 7), (1, 1), kernel_initializer=kernel_init, padding='same', use_bias=False)(
        input_data)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    x = activation_function(x)
    x = downsample(x, 128)
    x = downsample(x, 256)
    for _ in range(6):
        x = residual_block(x)
    x = upsample(x, 128)
    x = upsample(x, 64)
    x = tf.keras.layers.Conv2D(3, (7, 7), (1, 1), kernel_initializer=kernel_init, padding='same', use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    output_data = tf.keras.layers.Activation("tanh")(x)
    model = tf.keras.models.Model(input_data, output_data)
    # print(model.summary())
    return model
