#!/usr/bin/env python
# coding=utf-8

# @Author        :chenfeiyu
# @Company       :ByteDance
# @Time          :2021/10/21 12:13 上午
# @Project Name  :DeepLearn
# @File Name     :Main2.py
# @IDE           :PyCharm
# @Description   :
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 2
        optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(), ])

        self.generator = self.build_generator()

        z = tf.keras.layers.Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        validity = self.discriminator(img)

        self.combined = tf.keras.models.Model(z, validity)
        self.combined.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=optimizer)

    def build_generator(self):

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Dense(256, input_dim=self.latent_dim))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.Dense(512))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.Dense(1024))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.Dense(np.prod(self.img_shape), activation=tf.keras.activations.tanh))
        model.add(tf.keras.layers.Reshape(self.img_shape))

        model.summary()

        noise = tf.keras.layers.Input(shape=(self.latent_dim,))
        img = model(noise)

        return tf.keras.models.Model(noise, img)

    def build_discriminator(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=self.img_shape))
        model.add(tf.keras.layers.Dense(512))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))
        model.summary()
        img = tf.keras.layers.Input(shape=self.img_shape)
        validity = model(img)
        return tf.keras.models.Model(img, validity)

    def train(self, epochs, batch_size, sample_interval):
        (X_train, Y_train), (_, _) = tf.keras.datasets.mnist.load_data()
        X_train = X_train / 125.0 - 1.
        X_train = np.expand_dims(X_train, axis=3)
        dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(batch_size * 10).batch(batch_size=batch_size)
        for epoch in range(epochs):
            for batch, (imgs, labels) in enumerate(dataset):
                batch_size = imgs.shape[0]
                real = np.ones((batch_size, 1))
                fake = np.zeros((batch_size, 1))
                z = np.random.normal(0, 1, (batch_size, self.latent_dim))
                g_imgs = self.generator.predict(z)
                # d_loss = self.discriminator.train_on_batch(tf.concat((imgs, g_imgs),axis=0), tf.concat((real, fake),axis=0))
                d_loss_real = self.discriminator.train_on_batch(imgs, real)
                d_loss_fake = self.discriminator.train_on_batch(g_imgs, fake)
                d_loss = np.add(d_loss_real, d_loss_fake) / 2
                z = np.random.normal(0, 1, (batch_size, self.latent_dim))
                g_loss = self.combined.train_on_batch(z, real)
                print("%d %d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" % (epoch, batch, d_loss[0], 100 * d_loss[1], g_loss))
                if batch % sample_interval == 0:
                    self.sample_images(f'{epoch}_{batch}', save_img=True)

    def sample_images(self, name, save_img=False):
        n = 10
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))

        grid_x = norm.ppf(np.linspace(0.01, 0.99, n))
        grid_y = norm.ppf(np.linspace(0.01, 0.99, n))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.generator.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        if save_img:
            plt.savefig("images/%s.png" % name)
        plt.show()

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=10, batch_size=64, sample_interval=500)
