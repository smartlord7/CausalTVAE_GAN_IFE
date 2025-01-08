import logging
from collections import defaultdict
import numpy as np
import QIDLearningLib.metrics.data_privacy as dp
import QIDLearningLib.metrics.qid_specific as qid
from sklearn.ensemble import IsolationForest
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_contamination_levels(start, end, step):
    """Generates contamination levels."""
    levels = np.arange(start=start, stop=min(0.5, end + step), step=step)
    logging.info(f"Generated contamination levels: {levels}")
    return levels


def fit_isolation_forest(column_data, contamination, random_state):
    """Fits an Isolation Forest on a single column and returns the anomaly predictions."""
    isolation_forest = IsolationForest(contamination=contamination, random_state=random_state)
    column_data = column_data.reshape(-1, 1).astype(np.float32)
    isolation_forest.fit(clean_data(column_data))
    logging.debug(f"Isolation Forest fitted with contamination: {contamination}, random_state: {random_state}")
    return isolation_forest.predict(column_data)


def calculate_anomaly_counts(columns, reconstruction_errors, contamination, num_trials, total_trials, progress_queue=None):
    """Calculates anomaly counts for each column over multiple trials."""
    total_columns = len(columns)
    total_combinations = num_trials * total_columns
    anomaly_counts_list = []
    mean_anomalies_per_column = np.zeros(total_columns)

    completed_combinations = 0  # To track the number of completed combinations

    for trial in range(num_trials):
        trial_anomaly_counts = []
        for i in range(total_columns):
            column_data = reconstruction_errors[:, i]
            random_state = trial * 1000 + int(contamination * 100)
            anomalies = fit_isolation_forest(column_data, contamination, random_state=random_state)
            num_anomalies = np.sum(anomalies == -1)
            trial_anomaly_counts.append(num_anomalies)
            mean_anomalies_per_column[i] += num_anomalies

            # Log progress after processing each column
            completed_combinations += 1
            progress = (completed_combinations / total_combinations) * 100
            logging.info(f"Trial {trial + 1}/{num_trials}, Column {i + 1}/{total_columns}. Progress: {progress:.2f}%")

            if progress_queue:
                progress_queue.put(1 / total_trials * 100)

        anomaly_counts_list.append(trial_anomaly_counts)

    avg_anomaly_counts = np.mean(np.array(anomaly_counts_list), axis=0).reshape(-1, 1)
    mean_anomalies_per_column /= num_trials  # Average anomalies across trials

    logging.info(f"Average anomaly counts calculated: {avg_anomaly_counts.flatten()}")
    logging.info(f"Mean number of anomalies per column: {mean_anomalies_per_column}")
    return avg_anomaly_counts, mean_anomalies_per_column


def identify_quasi_identifiers(anomaly_counts, columns, contamination):
    """Identifies quasi-identifiers using Isolation Forest on anomaly counts."""
    scaler = StandardScaler()
    anomaly_counts = scaler.fit_transform(anomaly_counts)
    anomaly_counts = np.exp(anomaly_counts)
    isolation_forest_final = IsolationForest(contamination=contamination, random_state=int(contamination * 100))
    isolation_forest_final.fit(anomaly_counts)

    scores = isolation_forest_final.decision_function(anomaly_counts)
    quasi_identifier_flags = isolation_forest_final.predict(anomaly_counts)
    quasi_identifier_indices_ = np.where(quasi_identifier_flags == -1)[0]
    quasi_identifiers = [columns[idx] for idx in quasi_identifier_indices_]

    logging.info(f"Identified quasi-identifiers: {quasi_identifiers}")
    return quasi_identifiers, quasi_identifier_indices_, scores


def calculate_metric(dataset, quasi_identifiers):
    """Calculates a custom metric for quasi-identifiers."""
    metric_value = ((1 / dp.k_anonymity(dataset, quasi_identifiers).mean) *
                    (qid.separation(dataset, quasi_identifiers) / 100 +
                     qid.distinction(dataset, quasi_identifiers) / 100))
    logging.info(f"Calculated metric value: {metric_value} for quasi-identifiers: {quasi_identifiers}")
    return metric_value


def train_and_evaluate_isolation_forest(dataset, columns, reconstruction_errors, contamination, num_trials, total_trials,
                                        progress_queue=None):
    """Trains and evaluates Isolation Forests for a given contamination level."""
    logging.info(f"Starting Isolation Forest with contamination level: {contamination}")

    anomaly_counts, mean_anomalies_per_column = calculate_anomaly_counts(columns, reconstruction_errors, contamination, num_trials, total_trials, progress_queue)
    quasi_identifiers, quasi_identifier_indices_, scores = identify_quasi_identifiers(anomaly_counts, columns,
                                                                                      contamination)

    if not quasi_identifiers:
        logging.info(f"No quasi-identifiers found for contamination level: {contamination}")
        return None

    metric_value = calculate_metric(dataset, quasi_identifiers)
    return metric_value, contamination, anomaly_counts, mean_anomalies_per_column, scores, quasi_identifier_indices_


def train_isolation_forest_ensemble(dataset, columns, reconstruction_errors,
                                    contamination_start=0.05,
                                    contamination_end=0.5,
                                    contamination_step=0.02,
                                    num_trials=5,  # Number of trials for statistical relevance
                                    progress_queue=None):
    """Trains an ensemble of Isolation Forests across different contamination levels."""
    contamination_levels = generate_contamination_levels(contamination_start, contamination_end, contamination_step)
    reconstruction_errors = np.exp(reconstruction_errors) # prominence to high values rather than low

    results = []
    weighted_votes = defaultdict(float)

    total_trials = len(contamination_levels) * num_trials * len(columns)

    with ThreadPoolExecutor() as executor:
        logging.info("Starting parallel execution of Isolation Forests")
        futures = [
            executor.submit(train_and_evaluate_isolation_forest, dataset, columns, reconstruction_errors, contamination,
                            num_trials, total_trials, progress_queue)
            for contamination in contamination_levels]

        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue

            metric_value, contamination, anomalies, mean_anomalies, scores, quasi_identifier_indices = result
            results.append((metric_value, contamination, anomalies, mean_anomalies, scores))

            for idx in quasi_identifier_indices:
                weighted_votes[columns[idx]] += metric_value
                logging.debug(f"Weighted vote updated for column {columns[idx]}: {weighted_votes[columns[idx]]}")

    final_quasi_identifiers = determine_final_quasi_identifiers(weighted_votes)
    final_quasi_identifier_indices = [np.where(columns == col)[0][0] for col in final_quasi_identifiers]

    return final_quasi_identifier_indices, columns[final_quasi_identifier_indices], weighted_votes, results


def determine_final_quasi_identifiers(weighted_votes):
    """Determines final quasi-identifiers based on weighted votes."""
    weighted_votes_values = list(weighted_votes.values())
    mn = np.median(weighted_votes_values)
    logging.info(f"Threshold for determining quasi-identifiers: {mn}")
    final_quasi_identifiers = [col for col, weight in weighted_votes.items() if weight >= mn]
    logging.info(f"Final quasi-identifiers selected: {final_quasi_identifiers}")
    return final_quasi_identifiers


def clean_data(data):
    # Check for NaN values
    if np.any(np.isnan(data)):
        print("Data contains NaNs. Replacing with 0.")
        data = np.nan_to_num(data)  # Replace NaNs with 0

    # Check for infinite values
    if np.any(np.isinf(data)):
        print("Data contains infinite values. Replacing with finite value.")
        # Determine the maximum finite value for the dtype
        max_value = np.finfo(data.dtype).max if np.issubdtype(data.dtype, np.floating) else np.iinfo(data.dtype).max
        min_value = np.finfo(data.dtype).min if np.issubdtype(data.dtype, np.floating) else np.iinfo(data.dtype).min

        # Replace positive and negative infinity
        data[np.isposinf(data)] = max_value
        data[np.isneginf(data)] = min_value

    return data