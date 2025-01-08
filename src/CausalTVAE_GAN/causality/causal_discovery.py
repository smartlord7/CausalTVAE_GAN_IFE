import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.PC import pc
from scipy.sparse import csgraph

from preprocess import sample_data, reduce_dimensions


def combine_adj_matrices(adj_matrices):
    # Assuming all adj_matrices are of the same size and correspond to the same nodes
    # Stack them vertically and then horizontally to form a block matrix
    combined_matrix = np.sum(adj_matrices, axis=0)
    combined_matrix[combined_matrix > 1] = 1  # Ensure adjacency matrix is binary

    return combined_matrix


def preprocess_for_cd(X, corr_threshold=0.95, variance_threshold=1e-5):
    # Step 1: Remove highly correlated features
    corr_matrix = np.corrcoef(X, rowvar=False)
    upper_triangle = np.triu(corr_matrix, k=1)
    to_drop = [col for col in range(upper_triangle.shape[1]) if any(upper_triangle[:, col] > corr_threshold)]
    X_reduced = np.delete(X, to_drop, axis=1)

    # Step 2: Remove near-zero variance features
    variances = np.var(X_reduced, axis=0)
    to_keep = np.where(variances >= variance_threshold)[0]
    X_reduced = X_reduced[:, to_keep]

    return X_reduced


def postprocess_cd(adjacency_matrix: np.ndarray, allow_self_loops: bool = False, interpolate: bool = True, interpolation_scale_factor: float = 0.6) -> np.ndarray:
    result_matrix = adjacency_matrix

    if not allow_self_loops:
        np.fill_diagonal(adjacency_matrix, 0)

    if interpolate:
        num_nodes = adjacency_matrix.shape[0]
        result_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

        # Get coordinates of all cells with value 1
        ones_indices = np.argwhere(adjacency_matrix == 1)

        # Iterate through each cell in the matrix
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adjacency_matrix[i, j] == 1:
                    continue
                else:
                    # Compute Euclidean distances from (i, j) to all cells with value 1
                    distances = np.sqrt((ones_indices[:, 0] - i) ** 2 + (ones_indices[:, 1] - j) ** 2)

                    # Calculate inverse distances to use as weights, avoiding division by zero
                    with np.errstate(divide='ignore', invalid='ignore'):
                        inverse_distances = np.where(distances != 0, 1 / distances, 0)


                    # Compute the weighted average for the interpolated value
                    interpolated_value = np.sum(inverse_distances)  # This should be based on weights, not a direct sum
                    result_matrix[i, j] = interpolated_value

        # Clip values to be in the range [0, 1]
        result_matrix = result_matrix / np.max(result_matrix)
        result_matrix = interpolation_scale_factor * result_matrix
        result_matrix[adjacency_matrix == 1] = 1

    return result_matrix


def causal_discovery(X, method='ges', sample_size=None, n_components=None, num_jobs=5):
    # Optional: Dimensionality reduction
    if n_components:
        X = reduce_dimensions(X, n_components=n_components)

    # Optional: Sampling
    if sample_size and X.shape[0] > sample_size:
        X = sample_data(X, sample_size=sample_size)

    # Define a method to run the selected causal discovery algorithm
    def run_method(subset, method):
        if method == 'ges':

            return ges(subset)['G']
        elif method == 'pc':

            return pc(subset).G
        else:
            raise ValueError(f"Unsupported method: {method}")

    # Run the causal discovery algorithm in parallel if needed
    if num_jobs > 0:
        subsets = np.array_split(X, num_jobs)
        results = Parallel(n_jobs=num_jobs)(delayed(run_method)(subset, method) for subset in subsets)
        # Aggregate results from parallel runs (you'll need to define how to combine these graphs)
        return combine_adj_matrices([result.dpath for result in results])
    else:
        try:

            return run_method(X, method)

        except np.linalg.LinAlgError:
            X = preprocess_for_cd(X)

            return run_method(X, method)


def visualize_causal_graph(adjacency_matrix):
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", arrows=True)
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


def pad_adjacency_matrix(adj_matrix, target_size):
    current_size = adj_matrix.shape[0]
    padded_matrix = np.zeros((target_size, target_size))
    padded_matrix[:current_size, :current_size] = adj_matrix

    return padded_matrix
