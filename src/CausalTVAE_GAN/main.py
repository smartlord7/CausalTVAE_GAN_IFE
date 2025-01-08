import logging
import numpy as np
import pandas as pd
import matplotlib
from random import seed
import matplotlib.pyplot as plt
from evaluate import evaluate_model
from explainability.report import generate_explanation_report
from gan.discriminator import create_gan_discriminator
from gan.gan import create_gan_model
from metrics.metrics import estimate_privacy_parameters
from util.file import load_data_from_folder, load_metadata
from util.graph import plot_error_histogram, plot_causal_graph
from isolation_forest.isolation_forest import train_isolation_forest_ensemble
from preprocess import preprocess_data
from keras.src.utils import plot_model
from causality.causal_discovery import causal_discovery
from train import train_models
from vae.vae import create_vae_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def qid_pipeline():
    logging.info("Starting the main function")
    matplotlib.use('TkAgg')
    seed(0)
    logging.info("Random seed set to 0")

    datasets, headers = load_data_from_folder('datasets')
    logging.info(f"Total datasets to process: {len(datasets)}")

    results = []  # List to store results

    for i, (name, dataset) in enumerate(datasets):
        logging.info(f"Processing dataset {i + 1}/{len(datasets)}: {name}")
        name = name.replace('.data', '')

        # Check for adult.json file
        metadata = load_metadata('datasets\\' + name, name)
        columns_to_discard = metadata.get('columns_discard', [])

        X, _ = preprocess_data(dataset, discard_columns=columns_to_discard)
        logging.info(f"Preprocessing completed for dataset: {name}. Shape after preprocessing: {X.shape}")

        causal_graph = causal_discovery(X, 'ges', num_jobs=0).dpath
        logging.info(f"Causal discovery completed for dataset: {name}. Graph path: {causal_graph}")

        plot_causal_graph(causal_graph)
        logging.info(f"Causal graph visualization completed for dataset: {name}")
        plt.close()

        shape = X.shape
        vae = create_vae_model(shape, causal_graph, num_heads=4, ff_dim=64, dropout_rate=0.1, latent_dim=16, )
        vae_performance = vae.summary()  # Capture the VAE model summary for the report
        logging.info(f"VAE model created for dataset: {name}")

        discriminator = create_gan_discriminator(shape[0], dropout_rate=0.1, dense_units=(64, 32))
        logging.info(f"GAN discriminator created for dataset: {name}")

        gan = create_gan_model(vae, discriminator)
        logging.info(f"GAN model created for dataset: {name}")

        plot_model(vae, to_file='vae_model.png', show_shapes=True, show_layer_names=True)
        logging.info(f"VAE model plot saved for dataset: {name} as 'vae_model.png'")

        train_models(X, vae, gan, discriminator, epochs=1, batch_size=X.shape[0] // 10)
        logging.info(f"Training completed for VAE and GAN models on dataset: {name}")

        reconstruction_errors = evaluate_model(X, vae, plot=False)
        logging.info(f"Evaluation of VAE model completed for dataset: {name}. Reconstruction errors calculated")

        plot_error_histogram(reconstruction_errors, name)
        logging.info(f"Reconstruction error histogram plotted for dataset: {name}")

        columns = np.array([column for column in headers[i][1] if column not in columns_to_discard])
        logging.info(f"Columns identified for dataset: {name}: {columns}")

        final_quasi_identifiers, weighted_votes = train_isolation_forest_ensemble(dataset, columns,
                                                                                  reconstruction_errors,
                                                                                  contamination_start=0.05,
                                                                                  contamination_end=0.5,
                                                                                  contamination_step=0.02
                                                                                  )
        logging.info(f"Isolation Forest ensemble training completed for dataset: {name}")
        logging.info(f"Final quasi-identifiers for dataset {name}: {final_quasi_identifiers}")

        # Store results in the list
        results.append({
            'file_name': name,
            'header': columns,
            'quasi_identifiers': final_quasi_identifiers
        })

        # Generate explanation report with additional details
        logging.info(f"Generating explainability report dataset: {name}...")
        report = generate_explanation_report(name, final_quasi_identifiers, weighted_votes, reconstruction_errors,
                                             causal_graph)
        logging.info(f"Report: \n ")

    # Save the results to a CSV file
    df_results = pd.DataFrame(results)
    df_results.to_csv('results.csv', index=False)
    logging.info("Results saved to 'results.csv'")


def estimate_parameters():
    logging.info("Starting the main function")
    matplotlib.use('TkAgg')
    seed(0)
    logging.info("Random seed set to 0")

    datasets, headers = load_data_from_folder('datasets')
    logging.info(f"Total datasets to process: {len(datasets)}")

    results = []  # List to store results for each dataset

    for i, (name, dataset) in enumerate(datasets):
        logging.info(f"Processing dataset {i + 1}/{len(datasets)}: {name}")
        dataset, _ = preprocess_data(dataset, discard_columns=None, scale_features=False)

        name = name.replace('.data', '')  # Clean dataset name

        # Estimate privacy and anonymization parameters for this dataset
        parameters = estimate_privacy_parameters(dataset)

        # Add dataset name to the parameters dictionary for identification in the CSV
        parameters['dataset_name'] = name

        # Append the parameters dictionary to results
        results.append(parameters)

    # Convert the list of dictionaries to a pandas DataFrame
    results_df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    results_df.to_csv('privacy_parameters.csv', index=False)

    logging.info("Privacy parameters have been saved to privacy_parameters.csv")



if __name__ == "__main__":
    logging.info("Script started")
    estimate_parameters()
    logging.info("Script finished")
