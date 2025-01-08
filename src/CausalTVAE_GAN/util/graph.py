import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from networkx import adjacency_matrix


def plot_causal_graph(adj_matrix_pre, adj_matrix, column_names, save_path=""):
    G = nx.from_numpy_array(adj_matrix_pre, create_using=nx.DiGraph)

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Use shell layout for better visualization
    pos = nx.shell_layout(G)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot the causal graph
    ax1.set_title("Causal Graph")
    nx.draw(
        G, pos, with_labels=True, labels={i: column_names[i] for i in range(len(column_names))},
        node_size=700, node_color="skyblue", font_size=12, font_weight="bold",
        edge_color="gray", arrows=True, arrowsize=20, node_shape='o', ax=ax1
    )

    # Plot the adjacency matrix
    ax2.set_title("Adjacency Matrix")
    sns.heatmap(adj_matrix, cmap="RdYlGn", annot=True, fmt='.3f', xticklabels=column_names,
                yticklabels=column_names, ax=ax2, cbar=False)
    ax2.set_xlabel("Columns")
    ax2.set_ylabel("Columns")

    plt.tight_layout()
    if save_path != "":
        plt.savefig(save_path)
    else:
        plt.show()


def plot_error_histogram(errors, name):
    plt.figure()
    plt.hist(errors, bins=30)
    plt.title(f'Reconstruction errors for {name}')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def plot_error_histograms(errors, columns, path=""):
    # Number of columns (attributes) in the dataset
    num_columns = errors.shape[1]

    # Determine the number of rows and columns for the subplot grid
    n_cols = 3  # Number of columns in the grid
    n_rows = int(np.ceil(num_columns / n_cols))  # Number of rows needed

    # Create subplots with the specified grid layout
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, n_rows * 4))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot each histogram in its own subplot
    for i in range(num_columns):
        axes[i].hist(errors[:, i], bins=50, alpha=0.5, label=f'{columns[i]}')
        axes[i].set_title(f'Histogram of Reconstruction Errors for {columns[i]}')
        axes[i].set_xlabel('Reconstruction Error')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()

    # Hide any unused subplots
    for j in range(num_columns, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()  # Adjust subplots to fit into figure area

    # Save the figure if a path is provided
    if path:
        plt.savefig(path)

