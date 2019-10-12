import pathlib
import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from IPython import display

gpus = tf.config.experimental.list_physical_devices('GPU')

# Allocate GPU memory sparingly to avoid memory issues.
# This if statement can be removed for better graphics cards than my
# GeForce 930MX
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

BUFFER_SIZE = 16187
BATCH_SIZE = 32
EPOCHS = 256

data_dir1 = tf.keras.utils.get_file(
    origin='http://imagenet.stanford.edu/internal/car196/cars_train.tgz', fname='cars_train', untar=True)
data_dir1 = pathlib.Path(data_dir1)

data_dir2 = tf.keras.utils.get_file(
    origin='http://imagenet.stanford.edu/internal/car196/cars_test.tgz', fname='cars_test', untar=True)
data_dir2 = pathlib.Path(data_dir2)

list_ds1 = tf.data.Dataset.list_files(str(data_dir1/'*.jpg'))
list_ds2 = tf.data.Dataset.list_files(str(data_dir2/'*.jpg'))

list_ds = list_ds1.concatenate(list_ds2)


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = (img - 0.5) * 2

    desired_width = 64
    desired_height = 64

    img = tf.image.resize_with_pad(img, desired_height, desired_width)

    return img


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


train_images = list_ds.map(
    process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_images = train_images.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((4, 4, 1024)))
    assert model.output_shape == (None, 4, 4, 1024)

    model.add(layers.Conv2DTranspose(
        512, (5, 5), strides=(2, 2), padding="same", use_bias=False))
    assert model.output_shape == (None, 8, 8, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        256, (5, 5), strides=(2, 2), padding="same", use_bias=False))
    assert model.output_shape == (None, 16, 16, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        128, (5, 5), strides=(2, 2), padding="same", use_bias=False))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2),
                                     padding="same", use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2),
                            padding='same', input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy()


def discriminator_loss(real_output, fake_output):
    real_noisy_labels = tf.random.uniform(tf.shape(real_output), 0, 0.1)
    real_flip_mask = tf.dtypes.cast(tf.random.uniform(
        tf.shape(real_output), 0, 1) < 0.05, dtype=tf.dtypes.float32) * 0.9
    real_noisy_labels_flipped = real_noisy_labels + real_flip_mask

    fake_noisy_labels = tf.random.uniform(tf.shape(fake_output), 0.9, 1)
    fake_flip_mask = tf.dtypes.cast(tf.random.uniform(
        tf.shape(fake_output), 0, 1) < 0.05, dtype=tf.dtypes.float32) * 0.9
    fake_noisy_labels_flipped = fake_noisy_labels - fake_flip_mask

    real_loss = cross_entropy(real_noisy_labels_flipped, real_output)

    fake_loss = cross_entropy(fake_noisy_labels_flipped, fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.zeros_like(fake_output), fake_output)


generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss, tf.norm(gradients_of_generator[-1]), tf.norm(gradients_of_discriminator[-1]), tf.norm(gradients_of_generator[0]), tf.norm(gradients_of_discriminator[0])


summary_writer = tf.summary.create_file_writer('tensorboard')


def train(dataset, epochs):
    minibatch = 0
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            losses = (train_step(image_batch))
            with summary_writer.as_default():
                tf.summary.scalar('generator_loss', losses[0], step=minibatch)
                tf.summary.scalar('discriminator_loss',
                                  losses[1], step=minibatch)
                tf.summary.scalar('generator_gradient_top',
                                  losses[2], step=minibatch)
                tf.summary.scalar('discriminator_gradient_top',
                                  losses[3], step=minibatch)
                tf.summary.scalar('generator_gradient_bottom',
                                  losses[4], step=minibatch)
                tf.summary.scalar('discriminator_gradient_bottom',
                                  losses[5], step=minibatch)
            minibatch += 1

        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(
            epoch + 1, time.time() - start))

    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i])
        plt.axis('off')
    fig.savefig('image_at_epoch{:04d}.png'.format(epoch))
    # plt.show()


train(train_images, EPOCHS)
