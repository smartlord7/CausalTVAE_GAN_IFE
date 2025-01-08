import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def train_models(X_train, vae, gan, discriminator, epochs=20, batch_size=128, progress_queue=None):
    # Optimize dataset loading
    dataset = tf.data.Dataset.from_tensor_slices(X_train) \
        .shuffle(buffer_size=1024) \
        .batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE) \
        .cache()

    total_batches = len(X_train) // batch_size
    total_batches_per_epoch = total_batches
    total_batches_overall = total_batches * epochs

    d_losses = []
    gan_losses = []

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.set_xlim(0, epochs)
    ax.set_ylim(0, 1)  # Adjust based on expected loss range

    batch_counter = 0

    for epoch in range(epochs):
        d_loss_real_list = []
        d_loss_fake_list = []
        gan_loss_list = []

        for real_data in dataset:
            # Generate fake data using VAE
            generated_data = vae.predict(real_data)

            # Discriminator loss on real data
            d_loss_real = discriminator.train_on_batch(real_data, np.ones((len(real_data), 1)))
            # Discriminator loss on generated data
            d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((len(generated_data), 1)))
            # GAN loss
            gan_loss = gan.train_on_batch(generated_data, np.ones((len(generated_data), 1)))  # Use generated data

            d_loss_real_list.append(d_loss_real)
            d_loss_fake_list.append(d_loss_fake)
            gan_loss_list.append(gan_loss)

            # Update progress queue at each batch
            batch_counter += 1
            if progress_queue is not None:
                progress = min(float(100), (batch_counter / total_batches_overall) * 100)
                progress_queue.put(progress)

        # Average losses for the epoch
        avg_d_loss_real = np.mean(d_loss_real_list)
        avg_d_loss_fake = np.mean(d_loss_fake_list)
        avg_d_loss = 0.5 * (avg_d_loss_real + avg_d_loss_fake)
        avg_gan_loss = np.mean(gan_loss_list)

        d_losses.append(avg_d_loss)
        gan_losses.append(avg_gan_loss)

        # Update interactive plot
        ax.clear()
        ax.plot(range(1, epoch + 2), d_losses, label='Discriminator Loss')
        ax.plot(range(1, epoch + 2), gan_losses, label='GAN Loss')
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        plt.draw()

        print(f"Epoch {epoch + 1}/{epochs}, Discriminator Loss: {avg_d_loss}, GAN Loss: {avg_gan_loss}")

    plt.ioff()  # Turn off interactive mode
    plt.show()
    plt.close()

    return d_losses, gan_losses