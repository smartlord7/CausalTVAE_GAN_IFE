from keras import Model
from keras.src.layers import Dense, LeakyReLU, BatchNormalization, Dropout
from keras.src.optimizers import Adam


def create_gan_model(vae, discriminator,
                     gan_learning_rate=0.0005, gan_beta_1=0.9,
                     loss_function='binary_crossentropy',
                     additional_layers=True,
                     hidden_units=256, dropout_rate=0.3):
    # Define the input and output for the GAN model
    gan_input = vae.input
    vae_output = vae(gan_input)
    discriminator.trainable = False

    # Adding extra layers to make the GAN more sophisticated
    x = vae_output

    if additional_layers:
        x = Dense(hidden_units)(x)
        x = LeakyReLU(negative_slope=0.2)(x)  # Use negative_slope instead of alpha
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

        x = Dense(hidden_units // 2)(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

        # Ensure the output shape matches the input shape of the discriminator
        x = Dense(discriminator.input_shape[-1])(x)

    # Final output layer
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)

    # Define optimizers with parametrizable learning rates and beta_1 values
    gan_optimizer = Adam(learning_rate=gan_learning_rate, beta_1=gan_beta_1)

    # Compile models with the specified loss function
    gan.compile(optimizer=gan_optimizer, loss=loss_function)  # Compile the GAN

    return gan
