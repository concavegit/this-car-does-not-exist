import glob
import os
import time
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from IPython import display
import pathlib

from tensorflow.keras import layers

BUFFER_SIZE = 8144
BATCH_SIZE = 1024
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
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = (img - 0.5) * 2

    desired_width = 128
    desired_height = 96

    img = tf.image.resize_with_pad(img, desired_height, desired_width)

    return img


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


train_images = list_ds.map(
    process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_images = train_images.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

f, ax = plt.subplots(4, 4)
for i, img in enumerate(train_images.take(16)):
    ax[i // 4][i % 4].imshow(img.numpy()[0, :, :, 0])
    ax[i // 4][i % 4].axis('off')
plt.show()


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(6 * 8 * 1024, use_bias=False, input_shape=(512,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((6, 8, 1024)))
    assert model.output_shape == (None, 6, 8, 1024)

    model.add(layers.Conv2DTranspose(
        512, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 6, 8, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 12, 16, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 24, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 48, 64, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2),
                                     padding='same', use_bias=False,
                                     activation='tanh'))
    assert model.output_shape == (None, 96, 128, 1)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2),
                            padding='same', input_shape=[96, 128, 1]))

    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

noise_dim = 512
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


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

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
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    fig.savefig('image_at_epoch{:04d}.png'.format(epoch))
    # plt.show()


train(train_images, EPOCHS)
