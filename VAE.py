import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.stats import norm


class VAE(object):
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.latent_dim = 2 # 隐变量取2维只是为了方便后面画图
        self.intermediate_dim = 256
        self.vae = self.build_VAE()
        self.encoder = self.build_Encoder()
        self.decoder = self.build_Decoder()

    # 重参数技巧
    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.math.exp(z_log_var / 2) * epsilon

    def build_Encoder(self):
        encoder = tf.keras.models.Model(self.encoder_input, self.z_mean)
        return encoder

    def build_Decoder(self):
        decoder_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        _h_decoded = self.decoder_h(decoder_input)
        _x_decoded_mean = self.decoder_mean(_h_decoded)
        decoder = tf.keras.models.Model(decoder_input, _x_decoded_mean)
        return decoder

    def build_VAE(self):
        self.encoder_input = tf.keras.layers.Input(shape=(self.img_rows * self.img_cols,))
        h = tf.keras.layers.Dense(self.intermediate_dim, activation=tf.keras.activations.relu)(self.encoder_input)
        # 算p(Z|X)的均值和方差
        self.z_mean = tf.keras.layers.Dense(self.latent_dim)(h)
        self.z_log_var = tf.keras.layers.Dense(self.latent_dim)(h)
        # 重参数层，相当于给输入加入噪声
        z = tf.keras.layers.Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])
        # 解码层，也就是生成器部分
        self.decoder_h = tf.keras.layers.Dense(self.intermediate_dim, activation=tf.keras.activations.relu)
        self.decoder_mean = tf.keras.layers.Dense(self.img_rows * self.img_cols, activation=tf.keras.activations.sigmoid)
        h_decoded = self.decoder_h(z)
        x_decoded_mean = self.decoder_mean(h_decoded)

        # 建立模型
        vae = tf.keras.models.Model(self.encoder_input, x_decoded_mean)

        # xent_loss是重构loss，kl_loss是KL loss
        xent_loss = tf.keras.losses.binary_crossentropy(self.encoder_input, x_decoded_mean)
        # xent_loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(self.encoder_input, x_decoded_mean), axis=-1)
        kl_loss = - 0.5 * tf.math.reduce_sum(1 + self.z_log_var - tf.math.pow(self.z_mean, 2) - tf.math.exp(self.z_log_var), axis=-1)
        vae_loss = tf.math.reduce_mean(xent_loss + kl_loss)

        # add_loss是新增的方法，用于更灵活地添加各种loss
        vae.add_loss(vae_loss)
        vae.compile(optimizer=tf.keras.optimizers.RMSprop())
        vae.summary()
        return vae

    def train(self, epochs, batch_size, sample_interval):
        (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

        X_train = X_train / 255.
        X_test = X_test / 255.
        X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
        X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
        dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(batch_size * 10).batch(batch_size=batch_size)

        for epoch in range(epochs):
            for batch, (imgs, labels) in enumerate(dataset):
                g_loss = self.vae.train_on_batch(imgs)
                print(f"epoch:{epoch},batch:{batch},g_loss:{g_loss}")
                if batch % sample_interval == 0:
                    self.latent_distribution(X_test, Y_test, batch_size, f'{epoch}_{batch}',save_img=True)
                    self.sample_images(f'{epoch}_{batch}',save_img=True)

    def latent_distribution(self, X_test, Y_test, batch_size, name, save_img=False):
        x_test_encoded = self.encoder.predict(X_test, batch_size=batch_size)
        plt.figure(figsize=(6, 6))
        plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=Y_test)
        plt.colorbar()
        if save_img:
            plt.savefig("images/%s_dis.png" % name)
        plt.show()

    def sample_images(self, name, save_img=False):
        # 观察隐变量的两个维度变化是如何影响输出结果的
        n = 10  # figure with 15x15 digits
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))

        # 用正态分布的分位数来构建隐变量对
        grid_x = norm.ppf(np.linspace(0.01, 0.99, n))
        grid_y = norm.ppf(np.linspace(0.01, 0.99, n))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        if save_img:
            plt.savefig("images/%s.png" % name)
        plt.show()


if __name__ == '__main__':
    vae = VAE()
    vae.train(epochs=100, batch_size=64, sample_interval=500)
