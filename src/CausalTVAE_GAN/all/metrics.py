# Helper functions for calculating dataset properties
import logging

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import entropy, zscore
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import StandardScaler


def find_optimal_dbscan_params(scaled_data):
    # Calculate k-nearest neighbors
    k = 4  # We use k=4 to find the distance to the 4th nearest neighbor for epsilon estimation
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(scaled_data)
    distances, indices = neighbors_fit.kneighbors(scaled_data)

    # Get the distances to the k-th nearest neighbors
    distances = np.sort(distances[:, k-1], axis=0)  # distance to the 4th nearest neighbor
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('Knee Method for DBSCAN Epsilon Selection')
    plt.xlabel('Data Points sorted by distance to 4th nearest neighbor')
    plt.ylabel('Distance')
    plt.grid()
    plt.show()

    # Implementing the knee point detection
    # Using the second derivative method to find the elbow
    diff = np.diff(distances)
    diff2 = np.diff(diff)

    # Find the index of the knee point
    knee_index = np.argmin(diff2)
    optimal_eps = distances[knee_index]
    logging.info(f"Optimal epsilon found: {optimal_eps}")

    # Setting min_samples based on dataset size
    num_records = len(scaled_data)
    optimal_min_samples = max(5, num_records // 100)  # At least 5 or 1% of total records

    logging.info(f"Optimal min_samples set to: {optimal_min_samples}")

    return optimal_eps, optimal_min_samples

def apply_dbscan(scaled_data):
    optimal_eps, optimal_min_samples = find_optimal_dbscan_params(scaled_data)

    n_clusters = 0
    attempt = 0
    max_attempts = 10  # Maximum number of attempts to find clusters
    eps_factor = 1.5  # Factor to increase eps
    min_samples_factor = 0.5  # Factor to decrease min_samples

    while n_clusters == 0 and attempt < max_attempts:
        attempt += 1
        logging.info(f"Attempt {attempt}: Trying DBSCAN with eps={optimal_eps} and min_samples={optimal_min_samples}")

        # Try DBSCAN with the initially calculated parameters
        dbscan = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples).fit(scaled_data)
        n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        logging.info(f"Calculated Number of DBScan Clusters: {n_clusters}")

        if n_clusters == 0:
            logging.warning("No clusters found, adjusting parameters...")
            # Adjust eps and min_samples for the next attempt
            optimal_eps *= eps_factor  # Increase epsilon
            optimal_min_samples = max(1, int(optimal_min_samples * min_samples_factor))  # Decrease min_samples
            logging.info(f"Adjusted epsilon to: {optimal_eps}")
            logging.info(f"Adjusted min_samples to: {optimal_min_samples}")

    if n_clusters == 0:
        logging.error("Unable to find clusters after multiple attempts.")
        return 1
    else:
        logging.info(f"Final number of clusters found: {n_clusters}")

    return n_clusters

# Helper functions for calculating dataset properties
def calculate_entropy(dataset):
    """Calculate entropy for each column and return the average."""
    entropies = []
    for col in dataset.columns:
        prob = dataset[col].value_counts(normalize=True)
        entropies.append(entropy(prob, base=2))
    return np.mean(entropies)

def calculate_skewness(dataset):
    """Calculate skewness for each numeric column and return the average skewness."""
    return dataset.skew().mean()

def calculate_kurtosis(dataset):
    """Calculate kurtosis for each numeric column and return the average kurtosis."""
    return dataset.kurtosis().mean()

def calculate_outliers(dataset):
    """Calculate outliers based on Z-score for numeric data."""
    z_scores = np.abs(zscore(dataset.select_dtypes(include=[np.number])))
    outliers = (z_scores > 3).sum(axis=1)
    return outliers.mean()

def calculate_density(dataset):
    """Estimate density of data using Kernel Density Estimation."""
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
    kde.fit(dataset)
    return np.mean(np.exp(kde.score_samples(dataset)))

def calculate_variance(dataset):
    """Calculate variance for numeric columns and return the average variance."""
    return dataset.var().mean()

def calculate_pairwise_correlation(dataset):
    """Calculate the average pairwise correlation between attributes."""
    corr_matrix = dataset.corr()
    return corr_matrix.mean().mean()

def calculate_missing_values(dataset):
    """Calculate percentage of missing values in the dataset."""
    return dataset.isnull().mean().mean()

def calculate_pairwise_distances(dataset):
    """Calculate the average pairwise distance between rows of the dataset."""
    dist_matrix = pdist(dataset, metric='euclidean')
    return np.mean(dist_matrix)

def calculate_pca_components(dataset):
    """Perform PCA and return the number of components that explain 95% variance."""
    pca = PCA(0.95)
    pca.fit(dataset)
    return len(pca.components_)


# Main function for inferring privacy parameters
def estimate_privacy_parameters(dataset):
    # Standardize the dataset for clustering and KDE
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset.select_dtypes(include=[np.number]))

    # Dataset properties
    num_records = len(dataset)
    num_attributes = len(dataset.columns)
    cardinality = dataset.nunique().mean()  # Average number of unique values per attribute
    logging.info(f"Calculated Average Cardinality: {cardinality}")

    # Calculating various statistics and logging their values
    avg_entropy = calculate_entropy(dataset)
    logging.info(f"Calculated Average Entropy: {avg_entropy}")

    avg_skewness = calculate_skewness(dataset)
    logging.info(f"Calculated Average Skewness: {avg_skewness}")

    avg_kurtosis = calculate_kurtosis(dataset)
    logging.info(f"Calculated Average Kurtosis: {avg_kurtosis}")

    avg_outliers = calculate_outliers(dataset)
    logging.info(f"Calculated Average Outliers: {avg_outliers}")

    avg_density = calculate_density(scaled_data)
    logging.info(f"Calculated Average Density: {avg_density}")

    avg_variance = calculate_variance(dataset)
    logging.info(f"Calculated Average Variance: {avg_variance}")

    avg_correlation = calculate_pairwise_correlation(dataset)
    logging.info(f"Calculated Average Correlation: {avg_correlation}")

    avg_missing = calculate_missing_values(dataset)
    logging.info(f"Calculated Average Missing Values: {avg_missing}")

    avg_pairwise_distance = calculate_pairwise_distances(scaled_data)
    logging.info(f"Calculated Average Pairwise Distance: {avg_pairwise_distance}")

    num_pca_components = calculate_pca_components(scaled_data)
    logging.info(f"Calculated Number of PCA Components: {num_pca_components}")

    # Choose DBSCAN parameters
    eps, min_samples = find_optimal_dbscan_params(scaled_data)

    # Perform DBSCAN clustering
    n_clusters = apply_dbscan(scaled_data)
    logging.info(f"Calculated Number of DBScan Clusters: {n_clusters}")

    #### Privacy Models ####
    # 1. k-Anonymity (Samarati and Sweeney, 1998)
    logging.info("Estimating k-Anonymity...")
    k_anonymity = max(3, int(np.log2(max(1, n_clusters)) + np.log2(max(1, cardinality)) + num_pca_components))
    logging.info(f"Estimated k-Anonymity: {k_anonymity} (Clusters: {n_clusters}, Cardinality: {cardinality}, PCA Components: {num_pca_components})")

    # 2. l-Diversity (Machanavajjhala et al. 2007)
    logging.info("Estimating l-Diversity...")
    l_diversity = max(2, int(np.ceil(2 + (avg_entropy / 2) + avg_skewness / 2 + avg_kurtosis / 2)))
    logging.info(f"Estimated l-Diversity: {l_diversity} (Entropy: {avg_entropy}, Skewness: {avg_skewness}, Kurtosis: {avg_kurtosis})")

    # 3. t-Closeness (Li, Li, and Venkatasubramanian, 2007)
    logging.info("Estimating t-Closeness...")
    t_closeness = min(0.5, 1 / (avg_entropy + avg_variance + avg_density + 1e-10))  # Added small constant to avoid division by zero
    logging.info(f"Estimated t-Closeness: {t_closeness} (Entropy: {avg_entropy}, Variance: {avg_variance}, Density: {avg_density})")

    # 4. Differential Privacy (Dwork et al., 2006)
    logging.info("Estimating Differential Privacy parameters...")
    epsilon = max(0.1, 1 / (np.log2(num_records + 1) + avg_skewness + avg_correlation + 1e-10))  # Avoid negative/zero values
    logging.info(f"Estimated Epsilon: {epsilon} (Records: {num_records}, Skewness: {avg_skewness}, Correlation: {avg_correlation})")

    delta = max(1e-7, 1 / (num_records ** 2 + avg_outliers + avg_missing + 1e-10))  # Avoid negative/zero values
    logging.info(f"Estimated Delta: {delta} (Outliers: {avg_outliers}, Missing: {avg_missing})")

    # Budget: Set a reasonable range for the budget
    budget = min(1, max(0, num_records / (avg_outliers + 1e-10)))  # Ensure budget is between 0 and 1
    logging.info(f"Estimated Budget: {budget} (Records: {num_records}, Outliers: {avg_outliers})")

    # 5. k-Map (SaNGreeK model)
    logging.info("Estimating k-Map...")
    k_map = max(3, int(np.log2(max(1, n_clusters)) + num_pca_components))
    estimator = 'Poisson' if avg_entropy > 0.5 else 'Zero Truncated Poisson'
    logging.info(f"Estimated k-Map: {k_map} (Clusters: {n_clusters}, PCA Components: {num_pca_components}, Estimator: {estimator})")

    significance_level = 0.05 if estimator != 'None' else None
    logging.info(f"Significance Level for k-Map: {significance_level}")

    # 6. Delta Presence (Domingo-Ferrer et al., 2006)
    logging.info("Estimating Delta Presence...")
    lower_delta = 1 / (1 + avg_outliers + 1e-10)  # Avoid division by zero
    upper_delta = min(1, max(0, 1 - avg_entropy))  # Ensure upper delta is between 0 and 1
    logging.info(f"Estimated Delta Presence: Lower = {lower_delta}, Upper = {upper_delta} (Outliers: {avg_outliers}, Entropy: {avg_entropy})")

    # 7. Profitability (Fung, Wang, and Yu, 2010)
    logging.info("Estimating Profitability...")
    attacker_model = 'prosecutor' if avg_entropy > 0.7 else 'journalist'
    logging.info(f"Estimated Profitability Attacker Model: {attacker_model} (Entropy: {avg_entropy})")

    # 8. Average Re-identification Risk (El Emam et al., 2011)
    logging.info("Estimating Average Re-identification Risk...")
    reid_risk_threshold = min(0.5, 1 / (avg_correlation + avg_skewness + 1e-10))  # Avoid division by zero
    logging.info(f"Estimated Average Re-identification Risk Threshold: {reid_risk_threshold} (Correlation: {avg_correlation}, Skewness: {avg_skewness})")

    # 9. Population Uniqueness (Dankar et al., 2013)
    logging.info("Estimating Population Uniqueness...")
    pop_uniqueness_threshold = min(0.2, 1 / (avg_entropy + 1e-10))  # Avoid division by zero
    pop_model = 'Dankar' if avg_entropy < 0.5 else 'Zayatz'
    logging.info(f"Estimated Population Uniqueness: Threshold = {pop_uniqueness_threshold}, Model = {pop_model} (Entropy: {avg_entropy})")

    # 10. Sample Uniqueness (Sweeney, 2002; Dankar et al., 2013)
    logging.info("Estimating Sample Uniqueness...")
    sample_uniqueness_threshold = 0.3  # Fixed threshold as per requirements
    logging.info(f"Estimated Sample Uniqueness Threshold: {sample_uniqueness_threshold}")

    #### Anonymization Techniques ####
    # 1. Generalization (Samarati and Sweeney, 1998)
    logging.info("Estimating Generalization Level...")
    generalization_level = min(0.5, 1 / (np.log2(max(1, cardinality + 1)) + avg_entropy + num_pca_components + 1e-10))  # Added constant to avoid division by zero
    logging.info(f"Estimated Generalization Level: {generalization_level} (Cardinality: {cardinality}, Entropy: {avg_entropy}, PCA Components: {num_pca_components})")

    # 2. Masking (Aggarwal et al., 2012)
    logging.info("Estimating Masking Strength...")
    masking_strength = min(0.5, 1 / (avg_outliers + avg_skewness + avg_kurtosis + 1e-10))  # Added constant to avoid division by zero
    logging.info(f"Estimated Masking Strength: {masking_strength} (Outliers: {avg_outliers}, Skewness: {avg_skewness}, Kurtosis: {avg_kurtosis})")

    # 3. Perturbation (Rubenstein et al., 2012)
    logging.info("Estimating Perturbation Amount...")
    perturbation_amount = min(0.5, 1 / (avg_entropy + avg_density + avg_pairwise_distance + 1e-10))  # Added constant to avoid division by zero
    logging.info(f"Estimated Perturbation Amount: {perturbation_amount} (Entropy: {avg_entropy}, Density: {avg_density}, Pairwise Distance: {avg_pairwise_distance})")

    # Return all inferred parameters
    return {
        'k_anonymity': k_anonymity,
        'l_diversity': l_diversity,
        't_closeness': t_closeness,
        'epsilon': epsilon,
        'delta': delta,
        'budget': budget,
        'k_map': k_map,
        'k_map_estimator': estimator,
        'k_map_significance': significance_level,
        'delta_presence_lower': lower_delta,
        'delta_presence_upper': upper_delta,
        'profitability_attacker_model': attacker_model,
        'average_reid_risk_threshold': reid_risk_threshold,
        'pop_uniqueness_threshold': pop_uniqueness_threshold,
        'pop_uniqueness_model': pop_model,
        'sample_uniqueness_threshold': sample_uniqueness_threshold,
        'generalization_level': generalization_level,
        'masking_strength': masking_strength,
        'perturbation_amount': perturbation_amount
    }