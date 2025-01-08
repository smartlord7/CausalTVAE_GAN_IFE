import os
import logging
import numpy as np
import QIDLearningLib.metrics.data_privacy as dp
import QIDLearningLib.metrics.qid_specific as qid
from matplotlib import pyplot as plt

from util.graph import plot_causal_graph, plot_error_histograms


def generate_explanation_report(dataset, dataset_name, columns, quasi_identifiers_indices, weighted_votes, reconstruction_errors,
                                causal_graph_adjacency_pre, causal_graph_adjacency):
    qids = [columns[idx] for idx in quasi_identifiers_indices]

    # Create directories for storing images
    os.makedirs('reports/images', exist_ok=True)

    # Save causal graph image
    causal_graph_path = os.path.join('reports', 'images', f'{dataset_name}_causal_graph.png').replace("\\", "/")
    plot_causal_graph(causal_graph_adjacency_pre, causal_graph_adjacency, columns, causal_graph_path)
    plt.close()

    # Save reconstruction errors image (as histograms)
    reconstruction_errors_path = os.path.join('reports', 'images', f'{dataset_name}_reconstruction_errors.png').replace("\\", "/")
    plot_error_histograms(reconstruction_errors, columns, reconstruction_errors_path)
    plt.close()

    # Calculate causal importance for all attributes
    causal_importance_values = np.sum(causal_graph_adjacency, axis=1)

    # Determine thresholds for low, medium, and high based on the distribution
    low_threshold = np.percentile(causal_importance_values, 33)
    high_threshold = np.percentile(causal_importance_values, 66)

    def categorize_causal_importance(value):
        if value <= low_threshold:
            return "Low"
        elif value <= high_threshold:
            return "Medium"
        else:
            return "High"

    # Define HTML report content with enhanced styling and explanations
    report_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<title>Explanation Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; font-size: 18px; line-height: 1.6; background-color: #f4f4f4; margin: 20px; }",
        "h1 { color: #333333; font-size: 28px; }",
        "h2 { color: #666666; font-size: 24px; margin-top: 20px; }",
        "h3 { color: #444444; font-size: 20px; margin-top: 15px; }",
        "p { font-size: 18px; margin-bottom: 10px; }",
        "table { width: 100%; border-collapse: collapse; margin-top: 20px; }",
        "table, th, td { border: 1px solid #dddddd; padding: 10px; text-align: left; }",
        "th { background-color: #333333; color: #ffffff; }",
        "tr:nth-child(even) { background-color: #f9f9f9; }",
        "img { display: block; margin: 20px auto; max-width: 80%; height: auto; border: 1px solid #cccccc; padding: 10px; background-color: #ffffff; }",
        "ul { margin-top: 10px; }",
        "li { margin-bottom: 5px; }",
        ".highlight { background-color: #ffff99 !important; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Explanation Report for Dataset: {dataset_name}</h1>",
        "<p>This report provides a detailed explanation of how quasi-identifiers were identified in your dataset. Quasi-identifiers are attributes that, when combined, can potentially identify individuals. The report includes information on the methods used to detect these quasi-identifiers, their significance, and how various models and metrics contributed to this identification.</p>",

        "<h2>1. Quasi-Identifiers Selected</h2>",
        f"<p><strong>Attributes:</strong> {', '.join(qids)}</p>",

        "<h2>2. Neural Network Models</h2>",
        "<p>We employed several neural network models to analyze and identify quasi-identifiers. Here’s a detailed overview of each model and its role:</p>",
        "<ul>",
        "<li><strong>Variational Autoencoder (VAE):</strong> The VAE is a type of neural network designed to compress and then reconstruct data. It helps determine how well we can reconstruct each attribute from a simplified version. High reconstruction errors for an attribute suggest that it carries more unique or sensitive information.</li>",
        "<ul>",
        "<li><strong>Transformer Encoder with Multihead Attention:</strong> Within the VAE, we use a Transformer Encoder equipped with Multihead Attention mechanisms. This setup allows the model to simultaneously focus on different parts of the data and learn complex patterns and relationships. Multihead Attention enhances the model's ability to capture complex dependencies between attributes, improving reconstruction accuracy.</li>",
        "<li><strong>Causal Aggregation Layer:</strong> This layer incorporates causal information produced by causal discovery algorithms, such as GES (Greedy Equivalence Search) and PC (Peter-Clark), to understand and utilize the causal relationships between attributes. The Causal Aggregation Layer applies these causal relationships through an adjacency matrix to enhance the model's understanding of how different features interact and influence each other.</li>",
        "</ul>",
        "<li><strong>Generative Adversarial Network (GAN):</strong> The GAN consists of two components: a generator that creates synthetic data (in this case the VAE) and a discriminator that differentiates between real and synthetic data. This adversarial setup helps in producing realistic data samples, which are then analyzed to detect anomalies or unusual patterns that might reveal quasi-identifiers.</li>",
        "</ul>",

        "<h2>3. Reconstruction Errors</h2>",
        "<p>Reconstruction errors indicate how well we can recreate an attribute from its compressed version. Large errors suggest that the attribute is more unique and could be important for identifying individuals:</p>",
        f"<img src='{reconstruction_errors_path.replace('reports/', '')}' alt='Reconstruction Errors'>",
        "<p><strong>Legend:</strong> The histogram shows the distribution of reconstruction errors. Attributes with larger errors are likely to be more significant for identification.</p>",

        "<h2>4. Causal Graph Insights</h2>",
        "<p>The causal graph visualizes the influence relationships between attributes. Stronger connections suggest more significant relationships:</p>",
        f"<img src='{causal_graph_path.replace('reports/', '')}' alt='Causal Graph'>",
        "<p><strong>Legend:</strong> Lines in the graph represent the influence between attributes. Attributes with stronger connections (darker lines) might be more critical for identifying individuals.</p>",

        "<ul>",
        "<li>Attributes with more connections in the causal graph are likely to be significant in identifying individuals.</li>",
        "<li>The graph provides a visual representation of how attributes are interconnected, aiding in identifying key attributes.</li>",
        "</ul>",

        "<h2>5. Isolation Forest Analysis</h2>",
        "<p>Isolation Forests were used to identify anomalies and unusual attributes. The analysis helps determine which attributes are more likely to be unique and important:</p>",
        "<ul>",
        "<li><strong>Isolation Scores:</strong> Attributes with higher isolation scores are considered more unusual or unique.</li>",
        "<li><strong>Weighted Votes:</strong> Attributes with higher votes are deemed more significant for identification.</li>",
        "</ul>",
    ]

    # Adding detailed analysis for each attribute
    report_lines.extend([
        "<h2>7. Detailed Attribute Analysis</h2>",
        "<table>",
        "<tr><th>Attribute</th><th>Weighted Vote</th><th>Mean Reconstruction Error</th><th>Causal Importance</th></tr>"
    ])

    # Adding detailed analysis for all attributes
    for idx in range(len(columns)):
        attr_errors = reconstruction_errors[:, idx]
        causal_importance_value = causal_importance_values[idx]
        causal_importance_category = categorize_causal_importance(causal_importance_value)
        weighted_vote = weighted_votes.get(columns[idx], 0)
        row_class = "highlight" if columns[idx] in qids else ""
        report_lines.append(
            f"<tr class='{row_class}'><td>{columns[idx]}</td><td>{weighted_vote:.4f}</td><td>{np.mean(attr_errors):.4f}</td><td>{causal_importance_category} ({causal_importance_value:.4f})</td></tr>"
        )

    report_lines.extend([
        "</table>",
        "<h2>8. Privacy Metrics</h2>",
        "<p>We calculated several metrics to evaluate the privacy impact of the identified quasi-identifiers:</p>",
        f"<p><strong>Separation:</strong> {qid.separation(dataset, qids):.4f}</p>",
        f"<p><strong>Distinction:</strong> {qid.distinction(dataset, qids):.4f}</p>",
        f"<p><strong>K-anonymity:</strong> {dp.k_anonymity(dataset, qids).mean:.4f}</p>",
        "<h2>9. Conclusion</h2>",
        "<p>This report outlines the process of identifying quasi-identifiers using advanced models and techniques. We analyzed reconstruction errors, causal relationships, and anomaly scores to pinpoint attributes that could potentially identify individuals. This comprehensive approach ensures a robust understanding of the dataset’s privacy risks and helps in safeguarding personal information while maintaining data utility.</p>",
        "</body>",
        "</html>"
    ])

    # Save the report
    report = "\n".join(report_lines)
    report_path = os.path.join('reports', f'{dataset_name}_explanation_report.html')

    os.makedirs('reports', exist_ok=True)

    with open(report_path, 'w') as report_file:
        report_file.write(report)

    logging.info(f"Explanation report saved for dataset {dataset_name} as '{report_path}'")

    return report_path
