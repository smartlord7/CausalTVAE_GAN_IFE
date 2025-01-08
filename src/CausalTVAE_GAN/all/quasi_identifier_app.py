import csv
import json
import logging
import multiprocessing
import os
import tempfile
import threading
import webbrowser
from collections.abc import Iterable
from queue import Queue
from random import seed
import tensorflow as tf
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog, messagebox
from tkinter import ttk
from keras.src.utils import plot_model
from scipy.stats import ttest_rel

from evaluate import evaluate_model
from report import generate_explanation_report
from discriminator import create_gan_discriminator
from gan import create_gan_model
from isolation_forest import train_isolation_forest_ensemble
from preprocess import preprocess_data
from causal_discovery import causal_discovery, postprocess_cd
from train import train_models
from graph import plot_causal_graph
from QIDLearningLib.metrics import performance as perf
from graph import plot_error_histograms
from vae import create_vae_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QuasiIdentifierApp:
    def __init__(self, root):
        # TODO
        # Mostrar path ground truth
        # Tornar dimensões do VAE dinamicas consoante o dataset X
        # Dividir configurações em seccções de step
        # Permitir alterar interp scale factor
        # Permitir alterar lr
        # Mais granularidade na progress bar treino/analise
        # Data na entrada do historico
        # Remover warnings retracing/weights/end of sequence

        self.quasi_identifiers_indices = None
        seed(0)
        tf.random.set_seed(0)
        self.ground_truth = None
        self.quasi_identifiers = []
        self.weighted_votes = None
        self.input_dim = None
        self.X = None
        self.root = root
        self.root.title("Quasi-Identifier Detection Tool")

        self.history_file = "history.json"
        self.history = self.load_history()

        # Initial dataset-related variables
        self.dataset = None
        self.headers = None
        self.dataset_name = None

        # Counters for sessions
        self.process_counter = 0
        self.train_counter = 0
        self.analysis_counter = 0

        # Model-related variables
        self.vae = None
        self.gan = None
        self.discriminator = None
        self.reconstruction_errors = None
        self.causal_graph = None
        self.metrics = [perf.specificity, perf.fpr, perf.precision, perf.recall, perf.f1_score, perf.f2_score,
                        perf.jaccard_similarity, perf.dice_similarity, perf.accuracy]

        # Hyperparameters dictionary
        self.hyperparams = {
            'causal_algorithm': 'ges',
            'epochs': 20,
            'batch_size_factor': 10,
            'contamination_start': 0.2,
            'contamination_end': 0.5,
            'contamination_step': 0.02,
            'num_trials': 30
        }

        # Style

        # GUI Components
        self.create_widgets()

    def create_widgets(self):
        # Frame for file selection and data preview
        frame_top = Frame(self.root)
        frame_top.pack(pady=10, fill=X)

        self.label_file = Label(frame_top, text="Select Dataset:")
        self.label_file.pack(side=LEFT, padx=5)

        self.entry_file = Entry(frame_top, width=100)
        self.entry_file.pack(side=LEFT, padx=5)

        self.button_browse = Button(frame_top, text="Browse", command=self.load_dataset)
        self.button_browse.pack(side=LEFT, padx=5)

        # New button to open the dataset viewer
        self.button_view_dataset = Button(frame_top, text="View Dataset", command=self.open_dataset_viewer,
                                          state=DISABLED)
        self.button_view_dataset.pack(side=LEFT, padx=5)

        self.button_process = Button(frame_top, text="Process", command=self.process_dataset, state=DISABLED)
        self.button_process.pack(side=LEFT, padx=5)

        # Buttons for Plot VAE Model and Plot Causal Graph next to Process button
        self.button_plot_all_models = Button(frame_top, text="View Models", command=self.plot_all_models,
                                             state=DISABLED)
        self.button_plot_all_models.pack(side=LEFT, padx=5)

        self.button_plot_causal_graph = Button(frame_top, text="View Causal Info", command=self.plot_causal_graph,
                                               state=DISABLED)
        self.button_plot_causal_graph.pack(side=LEFT, padx=5)

        self.button_advanced = Button(frame_top, text="Advanced", command=self.open_advanced_dialog)
        self.button_advanced.pack(side=RIGHT, padx=5)

        # Frame for displaying dataset metadata
        frame_metadata = Frame(self.root)
        frame_metadata.pack(fill=X, padx=10, pady=5)

        self.label_metadata = Label(frame_metadata, text="Metadata:")
        self.label_metadata.pack(side=LEFT, padx=5)

        # Treeview to display dataset metadata as a DataFrame
        self.tree_metadata = ttk.Treeview(frame_metadata, columns=("Key", "Value"), show="headings")
        self.tree_metadata.heading("Key", text="Key")
        self.tree_metadata.heading("Value", text="Value")

        self.tree_metadata.pack(fill=X, expand=True)

        # Treeview for displaying the dataset columns with additional information
        frame_tree = Frame(self.root)
        frame_tree.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.label_tree = Label(frame_tree, text="Columns:")
        self.label_tree.pack(side=LEFT, padx=5)

        # Treeview for displaying the dataset columns with additional information
        columns = ("Column", "Type", "Range/Values", "Histogram")
        self.tree = ttk.Treeview(frame_tree, columns=columns, show='headings', selectmode='extended')

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor=CENTER)

        # Default tag configuration
        self.tree.tag_configure("unselected", background="lightcoral")
        self.tree.tag_configure("selected", background="lightgreen")

        # Make the Treeview expand
        self.tree.pack(fill=BOTH, expand=True)

        self.button_select_all = Button(frame_tree, text="Select All", command=self.select_all_columns)
        self.button_select_all.pack(side=LEFT, padx=5)

        self.button_deselect_all = Button(frame_tree, text="Deselect All", command=self.deselect_all_columns)
        self.button_deselect_all.pack(side=LEFT, padx=5)

        # Frame for analysis options and plotting
        frame_bottom = Frame(self.root)
        frame_bottom.pack(pady=10)

        self.button_train = Button(frame_bottom, text="Train", command=self.train_models, state=DISABLED)
        self.button_train.pack(side=LEFT, padx=5)

        # Add Plot Reconstruction Errors button next to Train
        self.button_plot_errors = Button(frame_bottom, text=" View Reconstruction Errors",
                                         command=self.plot_reconstruction_errors, state=DISABLED)
        self.button_plot_errors.pack(side=LEFT, padx=5)

        # Add a vertical bar separator between Train/Plot Errors and Run Analysis/View Report
        separator = ttk.Separator(frame_bottom, orient="vertical")
        separator.pack(side=LEFT, fill=Y, padx=10)

        self.button_run_analysis = Button(frame_bottom, text="Analyse", command=self.run_analysis, state=DISABLED)
        self.button_run_analysis.pack(side=LEFT, padx=5)

        # Add a vertical bar separator between Train/Plot Errors and Run Analysis/View Report
        separator = ttk.Separator(frame_bottom, orient="vertical")
        separator.pack(side=LEFT, fill=Y, padx=10)

        self.button_view_report = Button(frame_bottom, text="View Report", command=self.view_report, state=DISABLED)
        self.button_view_report.pack(side=LEFT, padx=5)

        # Add a button to load ground truth data
        self.button_load_gt = Button(frame_bottom, text="Load Ground Truth", command=self.load_ground_truth)
        self.button_load_gt.pack(side=LEFT, padx=5)

        # Add a button to compare detected quasi-identifiers against ground truth
        self.button_compare_qid = Button(frame_bottom, text="Test",
                                         command=self.compare_quasi_identifiers, state=DISABLED)
        self.button_compare_qid.pack(side=LEFT, padx=5)


        # Frame for history
        frame_history = Frame(self.root)
        frame_history.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.label_history = Label(frame_history, text="History:")
        self.label_history.pack(side=LEFT, padx=5)

        self.tree_history = ttk.Treeview(frame_history, columns=("Type", "Details"), show='headings')
        self.tree_history.heading("Type", text="Type")
        self.tree_history.heading("Details", text="Details")
        self.tree_history.pack(fill=BOTH, expand=True)

        # Button to clear history
        self.clear_button = Button(frame_history, text="Clear History", command=self.clear_history)
        self.clear_button.pack(side=LEFT, padx=5)

        self.update_history_view()

    def check_ready_for_comparison(self):
        if hasattr(self, 'ground_truth') and len(self.quasi_identifiers) > 0:
            self.button_compare_qid.config(state=NORMAL)

    def open_dataset_viewer(self):
        # Create a new top-level window for the dataset viewer
        dataset_viewer = Toplevel(self.root)
        dataset_viewer.title("Dataset Viewer")

        # Create a frame for the Treeview and Scrollbars
        frame_treeview = Frame(dataset_viewer)
        frame_treeview.pack(fill=BOTH, expand=True)

        # Create Treeview widget
        tree = ttk.Treeview(frame_treeview, show="headings")

        # Add columns to the Treeview
        tree["columns"] = list(self.dataset.columns)  # Assuming self.dataset is a pandas DataFrame

        for col in tree["columns"]:
            tree.heading(col, text=col)
            tree.column(col, anchor=CENTER)

        # Insert data into Treeview
        for _, row in self.dataset.iterrows():
            tree.insert("", "end", values=list(row))

        # Add vertical scrollbar
        vsb = Scrollbar(frame_treeview, orient=VERTICAL, command=tree.yview)
        vsb.pack(side=RIGHT, fill=Y)
        tree.configure(yscrollcommand=vsb.set)

        # Add horizontal scrollbar
        hsb = Scrollbar(frame_treeview, orient=HORIZONTAL, command=tree.xview)
        hsb.pack(side=BOTTOM, fill=X)
        tree.configure(xscrollcommand=hsb.set)

        # Pack the Treeview widget
        tree.pack(fill=BOTH, expand=True)

        # Optionally, make the window resizable
        dataset_viewer.geometry("800x600")
        dataset_viewer.resizable(True, True)

    # History functions
    def save_history(self):
        with open(self.history_file, 'w') as file:
            json.dump(self.history, file, indent=4)

    def load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as file:
                return json.load(file)
        else:
            return {}

    def clear_history(self):
        # Clear the Treeview
        for item in self.tree_history.get_children():
            self.tree_history.delete(item)

        # Delete the JSON file if it exists
        if os.path.exists(self.history_file):
            os.remove(self.history_file)

    def update_history(self, dataset_name, processes=None, train_sessions=None, analysis_results=None, details=None):
        # Add a dataset entry if not already present
        if dataset_name not in self.history:
            self.history[dataset_name] = []

        if processes:
            for process_name in processes:
                # Increment process counter
                self.process_counter += 1
                process_entry = f"  {process_name} (Process {self.process_counter})" if not details else str(details)
                self.history[dataset_name].append({"Type": (" " * 4) + "Process", "Details": (" " * 4) + process_entry})

                for analysis in processes[process_name]:
                    # Increment analysis counter
                    self.analysis_counter += 1
                    analysis_entry = f"      {analysis} (Analysis {self.analysis_counter})" if not details else str(
                        details)
                    self.history[dataset_name].append(
                        {"Type": (" " * 16) + "Analysis", "Details": (" " * 16) + analysis_entry})

        if train_sessions:
            for train_name in train_sessions:
                # Increment train counter
                self.train_counter += 1
                train_entry = f"    {train_name} (Training {self.train_counter})" if not details else str(details)
                self.history[dataset_name].append({"Type": (" " * 8) + "Training", "Details": (" " * 8) + train_entry})

        if analysis_results:
            for analysis_name in analysis_results:
                # Increment analysis counter
                self.analysis_counter += 1
                analysis_entry = f"      {analysis_name} (Analysis {self.analysis_counter})" if not details else str(
                    details)
                self.history[dataset_name].append(
                    {"Type": (" " * 16) + "Analysis", "Details": (" " * 16) + analysis_entry})

        self.save_history()
        self.update_history_view()

    def update_history_view(self):
        self.tree_history.delete(*self.tree_history.get_children())

        for dataset_name, entries in self.history.items():
            dataset_node = self.find_or_create_node(dataset_name, "Dataset")
            for entry in entries:
                self.tree_history.insert(dataset_node, "end", values=(entry["Type"], entry["Details"]))

    def find_or_create_node(self, name, node_type, parent=None):
        if parent:
            # Search for existing child node
            for child in self.tree_history.get_children(parent):
                if self.tree_history.item(child, "values")[0] == node_type and name in \
                        self.tree_history.item(child, "values")[1]:
                    return child
            # Create new node if not found
            return self.tree_history.insert(parent, "end", values=(node_type, name))
        else:
            # Create top-level node
            return self.tree_history.insert("", "end", values=(node_type, name))

    def populate_treeview(self, columns):
        self.tree.delete(*self.tree.get_children())
        for col in columns:
            col_type = str(self.dataset[col].dtype)
            col_range = self.get_column_range_or_values(col)
            self.tree.insert("", "end", iid=col, values=(col, col_type, col_range, "Plot"), tags=("selected",))

        # Add a Button to toggle selection when "Toggle" is clicked
        self.tree.bind("<ButtonRelease-1>", self.on_treeview_click)

    def get_column_range_or_values(self, col):
        if pd.api.types.is_numeric_dtype(self.dataset[col]):
            return f"{self.dataset[col].min()} - {self.dataset[col].max()}"
        elif pd.api.types.is_categorical_dtype(self.dataset[col]) or self.dataset[col].dtype == object:
            return ', '.join(self.dataset[col].unique().astype(str)) + '...'  # Show up to 10 unique values
        else:
            return "N/A"

    def on_treeview_click(self, event):
        item = self.tree.identify('item', event.x, event.y)
        column = self.tree.identify_column(event.x)

        if column == "#1":  # Assuming the column name is in the 1st column
            current_tags = self.tree.item(item, "tags")

            # Toggle selection
            if "unselected" in current_tags:
                new_tags = ("selected",)
                self.tree.item(item, tags=new_tags)
                self.tree.tag_configure("selected", background="lightgreen")
            else:
                new_tags = ("unselected",)
                self.tree.item(item, tags=new_tags)
                self.tree.tag_configure("unselected", background="lightcoral")
        elif column == '#4':  # Assuming 'Histogram' is the 5th column
            col_name = self.tree.item(item, "values")[0]
            self.plot_column_histogram(col_name)

    def select_all_columns(self):
        for item in self.tree.get_children():
            self.tree.item(item, tags=("selected",))
            self.tree.tag_configure("selected", background="lightgreen")

    def deselect_all_columns(self):
        for item in self.tree.get_children():
            self.tree.item(item, tags=("unselected",))
            self.tree.tag_configure("unselected", background="lightcoral")

    def plot_column_histogram(self, col_name):
        plt.figure()
        self.dataset[col_name].hist(bins=30)
        plt.title(f'Histogram of {col_name}')
        plt.xlabel(col_name)
        plt.ylabel('Frequency')
        plt.show()

    def load_ground_truth(self):
        file_selected = filedialog.askopenfilename(
            title="Select Ground Truth",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_selected:
            try:
                self.ground_truth = pd.read_csv(file_selected)
                messagebox.showinfo("Ground Truth Loaded", "Loaded ground truth data successfully.")
                if self.dataset_name:
                    self.button_compare_qid.config(state=NORMAL)
            except Exception as e:
                logging.error(f"Error loading ground truth data: {e}")
                messagebox.showerror("Error", "Failed to load the ground truth data.")

    def compare_quasi_identifiers(self):
        # Create a new top-level window to show comparison and anonymization results
        self.comparison_window = Toplevel(self.root)
        self.comparison_window.title("Comparison and Anonymization Results")

        if not hasattr(self, 'ground_truth'):
            messagebox.showerror("Error", "Ground truth data not loaded.")
            return

        detected_dataset = self.dataset_name.replace(".data", "")

        # Assuming the ground truth is a DataFrame with a similar structure
        self.ground_truth_qids = self.ground_truth[self.ground_truth['dataset'] == detected_dataset]['qids']

        # Compare detected quasi-identifiers with ground truth
        self.show_comparison_results()

        # Create a Frame for the anonymization testing part
        anonymization_frame = Frame(self.comparison_window)
        anonymization_frame.pack(fill=BOTH, expand=True)

        Label(anonymization_frame, text="Privacy Model:").grid(row=0, column=0, padx=10, pady=5)
        self.combo_privacy_model = ttk.Combobox(anonymization_frame, values=["kanonymity"], state="readonly", width=47)
        self.combo_privacy_model.grid(row=0, column=1, padx=10, pady=5)
        self.combo_privacy_model.current(0)

        Label(anonymization_frame, text="Initial k-Value:").grid(row=1, column=0, padx=10, pady=5)
        self.entry_initial_k_value = Entry(anonymization_frame, width=50)
        self.entry_initial_k_value.grid(row=1, column=1, padx=10, pady=5)
        self.entry_initial_k_value.insert(0, "5")  # Default initial k-value

        Label(anonymization_frame, text="k-Step:").grid(row=2, column=0, padx=10, pady=5)
        self.entry_k_step = Entry(anonymization_frame, width=50)
        self.entry_k_step.grid(row=2, column=1, padx=10, pady=5)
        self.entry_k_step.insert(0, "1")  # Default step size

        Label(anonymization_frame, text="Final k-Value:").grid(row=3, column=0, padx=10, pady=5)
        self.entry_final_k_value = Entry(anonymization_frame, width=50)
        self.entry_final_k_value.grid(row=3, column=1, padx=10, pady=5)
        self.entry_final_k_value.insert(0, "10")  # Default final k-value

        Label(anonymization_frame, text="Suppression Limit:").grid(row=4, column=0, padx=10, pady=5)
        self.entry_suppression_limit = Entry(anonymization_frame, width=50)
        self.entry_suppression_limit.grid(row=4, column=1, padx=10, pady=5)
        self.entry_suppression_limit.insert(0, "0.05")  # Default suppression limit (5%)

        Label(anonymization_frame, text="Metric:").grid(row=5, column=0, padx=10, pady=5)
        self.combo_metric = ttk.Combobox(anonymization_frame, values=["entropy", "precision"], state="readonly",
                                         width=47)
        self.combo_metric.grid(row=5, column=1, padx=10, pady=5)
        self.combo_metric.current(0)

        # Create the Run button
        self.run_button = Button(anonymization_frame, text="Run Anonymization", command=self.run_anonymization)
        self.run_button.grid(row=6, column=0, columnspan=2, pady=20)

        # Frame for the "Close" and "Copy" buttons
        button_frame = Frame(self.comparison_window)
        button_frame.pack(pady=10)

        # Add "Close" button to the frame
        close_button = Button(button_frame, text="Close", command=self.comparison_window.destroy)
        close_button.pack(side="left", padx=5)

        # Add "Copy" button to the frame
        copy_button = Button(button_frame, text="Copy", command=self.copy_results_tables)
        copy_button.pack(side="left", padx=5)

    def copy_results_tables(self):
        self.copy_table_to_clipboard(self.metrics_table, "Comparison")
        messagebox.showinfo("Copy to Clipboard", "Results copied to clipboard!")

    def copy_table_to_clipboard(self, table, title, first=True):
        # Prepare the text to be copied
        copy_text = title + "\n"
        headers = [table.heading(col)["text"] for col in table["columns"]]
        copy_text += "\t".join(headers) + "\n"

        for row in table.get_children():
            values = table.item(row)["values"]
            copy_text += "\t".join(map(str, values)) + "\n"

        # Copy to clipboard
        if first:
            self.root.clipboard_clear()
        self.root.clipboard_append(copy_text)
        self.root.update()  # Keep the clipboard data available

    def run_anonymization(self):
        predicted_qids = self.predicted_qids_var.get().strip().replace(" ", "").split(",")
        ground_truth_qids = self.ground_truth_qids.array[0].replace("[", "").replace("]", "").replace(" ", "").split(",")

        # Retrieve user inputs and calculate k range
        initial_k = int(self.entry_initial_k_value.get())
        k_step = int(self.entry_k_step.get())
        final_k = int(self.entry_final_k_value.get())
        k_range = np.arange(start=initial_k, stop=final_k + 1, step=k_step)

        # Initialize dictionaries to store metrics data
        predicted_metrics_data = {}
        ground_truth_metrics_data = {}

        for k in k_range:
            # Run anonymization for predicted QIDs and store the results
            results_predicted_dict = self.run_anonymization_(predicted_qids, k)
            for metric, (before_values, after_values, diff_values) in results_predicted_dict.items():
                if metric not in predicted_metrics_data:
                    predicted_metrics_data[metric] = ([], [], [])
                predicted_metrics_data[metric][0].append(before_values)
                predicted_metrics_data[metric][1].append(after_values)
                predicted_metrics_data[metric][2].append(diff_values)

            # Run anonymization for ground truth QIDs and store the results
            results_ground_truth_dict = self.run_anonymization_(ground_truth_qids, k)
            for metric, (before_values, after_values, diff_values) in results_ground_truth_dict.items():
                if metric not in ground_truth_metrics_data:
                    ground_truth_metrics_data[metric] = ([], [], [])
                ground_truth_metrics_data[metric][0].append(before_values)
                ground_truth_metrics_data[metric][1].append(after_values)
                ground_truth_metrics_data[metric][2].append(diff_values)

        # Plot the metrics for both predicted and ground truth data
        self.plot_metrics(k_range, predicted_metrics_data, ground_truth_metrics_data)

        # Perform statistical tests and show results
        self.perform_statistical_tests(predicted_metrics_data, ground_truth_metrics_data)

    def plot_metrics(self, k_range, predicted_metrics_data, ground_truth_metrics_data):
        # Divide metrics into two groups
        risk_and_utility_metrics = [metric for metric in predicted_metrics_data if
                                    "Prosecutor" not in metric and "Journalist" not in metric and "Marketer" not in metric]
        attack_metrics = [metric for metric in predicted_metrics_data if
                          "Prosecutor" in metric or "Journalist" in metric or "Marketer" in metric]

        def create_figure(metrics, title_prefix, n_cols):
            num_metrics = len(metrics)
            num_cols = n_cols  # Number of columns in the plot grid
            num_rows = int(np.ceil(num_metrics / num_cols))  # Number of rows based on metrics and columns

            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), squeeze=False)
            fig.suptitle(f"{title_prefix}", fontsize=16)

            axs = axs.flatten()

            for i, metric in enumerate(metrics):
                predicted_data = predicted_metrics_data[metric]
                ground_truth_data = ground_truth_metrics_data[metric]

                # Extract after values
                predicted_after = np.array(predicted_data[1]).flatten()
                ground_truth_after = np.array(ground_truth_data[1]).flatten()

                # Plot After values
                axs[i].plot(k_range, predicted_after, label="Predicted", linestyle='--')
                axs[i].plot(k_range, ground_truth_after, label="Ground Truth")
                axs[i].set_title(f"{metric} - After")  # Title for subplot
                axs[i].set_xlabel("k")
                axs[i].set_ylabel(f"Metric")
                axs[i].legend()
                axs[i].grid(True)

            # Hide any unused subplots
            for j in range(i + 1, len(axs)):
                axs[j].axis('off')

            # Adjust layout and display the plots
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

        # Create figures for different types of metrics
        create_figure(risk_and_utility_metrics, "Re-identification Risk and Data Utility", 2)
        create_figure(attack_metrics, "Attack Models Risks", 3)


    def run_anonymization_(self, qids, k_value):
        # Retrieve user inputs from the entry widgets
        privacy_model = self.combo_privacy_model.get()
        suppression_limit = self.entry_suppression_limit.get()
        metric_type = self.combo_metric.get()

        # Check if all required fields are filled
        if not (privacy_model and k_value and suppression_limit and metric_type):
            messagebox.showerror("Input Error", "All fields must be filled out.")
            return

        try:
            k_value = int(k_value)
            suppression_limit = float(suppression_limit)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for k-value and suppression limit.")
            return

        # Absolute path to the JAR file
        abs_path = os.path.abspath("ARXAnonymizationTester_jar/ARXAnonymizationTester.jar")

        # Construct the command to run
        java_command = (
            f'java -jar "{abs_path}" '
            f'"{os.path.abspath(f"../datasets/{self.dataset_name}/{self.dataset_name}.csv")}" '
            f'{",".join(qids)} '
            f'{privacy_model} '
            f'{str(k_value)} '
            f'{str(suppression_limit)} '
            f'{metric_type}'
        )

        try:
            # Create temporary files to capture stdout and stderr
            with tempfile.NamedTemporaryFile(delete=False, mode='w+', encoding='latin-1', errors='ignore') as stdout_file, \
                    tempfile.NamedTemporaryFile(delete=False, mode='w+', encoding='latin-1', errors='ignore') as stderr_file:

                # Get file paths
                stdout_path = stdout_file.name
                stderr_path = stderr_file.name

                # Close the files to release handles
                stdout_file.close()
                stderr_file.close()

                # Run the command
                os.system(f'{java_command} > "{stdout_path}" 2> "{stderr_path}"')

            # Read the captured output
            with open(stdout_path, 'r', encoding='latin-1', errors='ignore') as file:
                stdout = file.read()
            with open(stderr_path, 'r', encoding='latin-1', errors='ignore') as file:
                stderr = file.read()

            if stderr:
                print("Standard Error:")
                print(stderr)

            return self.parse_results(stdout)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def parse_results(self, results):
        # Initialize a dictionary to store metric names with their before and after values
        metrics_data = {}

        # Process the results to populate the dictionary
        for line in results.splitlines():
            if line.strip():  # Ensure it's not an empty line
                # Split the line by commas
                values = line.split(',')
                if len(values) == 3:
                    try:
                        metric_name = values[0].strip()
                        before_value = float(values[1].strip('%'))
                        after_value = float(values[2].strip('%'))

                        # If the metric name is not already in the dictionary, initialize it
                        if metric_name not in metrics_data:
                            metrics_data[metric_name] = ([], [], [])

                        # Append the before and after values to the corresponding lists
                        metrics_data[metric_name][0].append(before_value)  # Before values
                        metrics_data[metric_name][1].append(after_value)  # After values
                        metrics_data[metric_name][2].append(after_value - before_value)  # Difference values

                    except ValueError:
                        print(f"Skipping line with non-numeric values: {line}")
                else:
                    print(f"Skipping malformed line: {line}")

        # Return the populated dictionary
        return metrics_data

    def perform_statistical_tests(self, predicted_metrics_data, ground_truth_metrics_data):
        # Create a new top-level window for statistical results
        stats_window = Toplevel(self.root)
        stats_window.title("Statistical Test Results")

        # Create a frame for the Treeview and scrollbar
        frame = Frame(stats_window)
        frame.pack(fill=BOTH, expand=True)

        # Create a Treeview widget for "After" metrics
        columns_after = ["Metric", "Mean Predicted After", "Mean Ground Truth After",
                         "p-value (After)", "Comparison After"]

        tree_after = ttk.Treeview(frame, columns=columns_after, show='headings')
        tree_after.pack(side=LEFT, fill=BOTH, expand=True)

        # Define column headings
        for col in columns_after:
            tree_after.heading(col, text=col)
            tree_after.column(col, width=150, anchor='center')

        # Create a vertical scrollbar linked to the Treeview
        scrollbar_after = Scrollbar(frame, orient="vertical", command=tree_after.yview)
        scrollbar_after.pack(side=RIGHT, fill=Y)
        tree_after.configure(yscrollcommand=scrollbar_after.set)

        # Create a frame for the "Before" metrics Treeview
        frame_before = Frame(stats_window)
        frame_before.pack(fill=BOTH, expand=True)

        # Create a Treeview widget for "Before" metrics
        columns_before = ["Metric", "Predicted Before", "Ground Truth Before"]

        tree_before = ttk.Treeview(frame_before, columns=columns_before, show='headings')
        tree_before.pack(side=LEFT, fill=BOTH, expand=True)

        # Define column headings
        for col in columns_before:
            tree_before.heading(col, text=col)
            tree_before.column(col, width=150, anchor='center')

        # Create a vertical scrollbar linked to the Treeview
        scrollbar_before = Scrollbar(frame_before, orient="vertical", command=tree_before.yview)
        scrollbar_before.pack(side=RIGHT, fill=Y)
        tree_before.configure(yscrollcommand=scrollbar_before.set)

        # Populate the Treeview with "Before" and "After" statistical results
        for metric in predicted_metrics_data:
            predicted_after = np.array(predicted_metrics_data[metric][1])
            ground_truth_after = np.array(ground_truth_metrics_data[metric][1])

            # Calculate means
            mean_predicted_after = np.mean(predicted_after)
            mean_ground_truth_after = np.mean(ground_truth_after)

            # Perform paired t-tests (after only)
            result_after = ttest_rel(predicted_after, ground_truth_after)

            p_after = result_after.pvalue[0]
            t_after = result_after.statistic[0]

            # Determine comparison results
            comparison_after = "Higher" if p_after < 0.05 and mean_predicted_after > mean_ground_truth_after \
                else "Lower" if p_after < 0.05 and mean_predicted_after < mean_ground_truth_after else "Not Significant"

            # Insert "After" results into Treeview
            tree_after.insert("", END, values=(
                metric,
                f"{mean_predicted_after:.4f}",
                f"{mean_ground_truth_after:.4f}",
                f"{p_after:.4f}",
                comparison_after
            ))

            # Extract "Before" values (first value from the array as they are the same)
            predicted_before = predicted_metrics_data[metric][0][0][0]
            ground_truth_before = ground_truth_metrics_data[metric][0][0][0]

            # Insert "Before" results into Treeview
            tree_before.insert("", END, values=(
                metric,
                f"{predicted_before:.4f}",
                f"{ground_truth_before:.4f}"
            ))

        # Add a Save as CSV button
        def save_as_csv():
            # Ask user for file name and location
            file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                     filetypes=[("CSV files", "*.csv")],
                                                     title="Save as CSV")
            if not file_path:
                return

            # Write data to CSV
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)

                # Write "After" metrics
                writer.writerow(columns_after)  # Write header row for "After" metrics
                for row in tree_after.get_children():
                    values = tree_after.item(row, 'values')
                    writer.writerow(values)  # Write data rows for "After" metrics

                writer.writerow([])  # Add an empty row

                # Write "Before" metrics
                writer.writerow(columns_before)  # Write header row for "Before" metrics
                for row in tree_before.get_children():
                    values = tree_before.item(row, 'values')
                    writer.writerow(values)  # Write data rows for "Before" metrics

            messagebox.showinfo("Export Complete", "Statistical results have been saved as CSV.")

        save_button = Button(stats_window, text="Save as CSV", command=save_as_csv)
        save_button.pack(pady=10)

        # Adjust layout
        frame.pack_propagate(False)
        frame_before.pack_propagate(False)
        stats_window.geometry("1200x600")  # Set a reasonable default size for the window

    def compare_qids(self, predicted_qids, real_qids, metrics):
        results = {}

        if predicted_qids is None or len(predicted_qids) == 0:
            return results

        predicted_qids = set(map(lambda x: x.strip(), predicted_qids))
        ground_truth_qids = set(map(lambda x: x.strip(), real_qids.array[0].replace("[", "").replace("]", "").split(",")))
        for metric in metrics:
            results[metric.__name__] = metric(predicted_qids, ground_truth_qids)
        return results

    def show_comparison_results(self):
        # Create a Frame to hold the widgets
        frame = Frame(self.comparison_window)
        frame.pack(fill=BOTH, expand=True)

        # Display the predicted and ground truth QIDs
        real_qids_str = ", ".join(self.ground_truth_qids)

        # Display ground truth and predicted QIDs
        Label(frame, text="Predicted QIDs:").pack(anchor="w", padx=10, pady=(10, 0))
        self.predicted_qids_var = StringVar(value=",".join(self.quasi_identifiers))
        self.predicted_qids_entry = Entry(frame, justify="left", textvariable=self.predicted_qids_var)
        self.predicted_qids_entry.pack(anchor="w", padx=10)

        Label(frame, text="Ground Truth QIDs:").pack(anchor="w", padx=10, pady=(10, 0))
        real_qids_label = Label(frame, text=real_qids_str, wraplength=600, justify="left")
        real_qids_label.pack(anchor="w", padx=10)

        # Create a Treeview to display the metric results
        columns = ("Metric", "Value")
        self.metrics_table = ttk.Treeview(frame, columns=columns, show='headings')
        for col in columns:
            self.metrics_table.heading(col, text=col)
            self.metrics_table.column(col, anchor=CENTER)

        self.metrics_table.pack(fill=BOTH, expand=True)

        def update():
            predicted_qids = self.predicted_qids_var.get().split(',')

            results = self.compare_qids(predicted_qids, self.ground_truth_qids, self.metrics)

            # Populate the Treeview with metric results
            def clear_table(tree):
                for row in tree.get_children():
                    tree.delete(row)

            clear_table(self.metrics_table)
            metrics_str = ""
            for metric_name, value in results.items():
                formatted_value = f"{value:.2f}"
                self.metrics_table.insert("", "end", values=(metric_name, formatted_value))
                metrics_str += f"{metric_name}: {formatted_value}\n"

        button_update = Button(frame, text="Update", command=update, state=NORMAL)
        button_update.pack(side=RIGHT, padx=5)

        update()

    def load_dataset(self):
        file_selected = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[("Data files", "*.data"), ("All files", "*.*")]
        )
        if file_selected:
            self.entry_file.delete(0, END)
            self.entry_file.insert(0, file_selected)

            try:
                self.dataset = pd.read_csv(file_selected, delimiter=',')
                self.dataset_name = os.path.basename(file_selected).replace('.data', '')
                self.headers = self.dataset.columns.tolist()
                self.populate_treeview(self.headers)
                self.button_process.config(state=NORMAL)
                self.button_view_dataset.config(state=NORMAL)

                # Extract metadata from the dataset
                metadata = {
                    "Number of Rows": len(self.dataset),
                    "Number of Columns": len(self.dataset.columns),
                    "Data Types Count": self.dataset.dtypes.nunique(),
                    "Missing Values": self.dataset.isnull().sum().sum(),
                    "Duplicate Rows": self.dataset.duplicated().sum(),
                    "Unique Rows": len(self.dataset.drop_duplicates()),
                    "Unique Values": self.dataset.nunique().sum(),
                    "Constant Columns": self.dataset.columns[self.dataset.nunique() == 1].tolist(),
                    "Entropy": f"{self.dataset.apply(lambda x: -np.sum((x.value_counts(normalize=True) * np.log2(x.value_counts(normalize=True)))), axis=0).sum():.3f}",
                    "Memory Usage (MB)": f"{self.dataset.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB",

                }

                # Clear the previous metadata
                for item in self.tree_metadata.get_children():
                    self.tree_metadata.delete(item)

                # Insert the new metadata
                for key, value in metadata.items():
                    self.tree_metadata.insert("", "end", values=(key, value))

                self.quasi_identifiers = []

                self.reconstruction_errors = None
                self.causal_graph = None
                self.button_plot_causal_graph.config(state=DISABLED)
                self.button_plot_all_models.config(state=DISABLED)
                self.button_train.config(state=DISABLED)
                self.button_plot_errors.config(state=DISABLED)
                self.button_run_analysis.config(state=DISABLED)
                self.button_view_report.config(state=DISABLED)
                self.button_compare_qid.config(state=DISABLED)

                if self.ground_truth is not None:
                    self.button_compare_qid.config(state=NORMAL)

                messagebox.showinfo("Dataset Loaded", f"Loaded dataset: {self.dataset_name}")

            except Exception as e:
                logging.error(f"Error loading dataset: {e}")
                messagebox.showerror("Error", "Failed to load the selected dataset.")

    def process_dataset(self):
        seed(0)
        tf.random.set_seed(0)
        # Collect selected columns based on the background color
        selected_columns = []
        for item in self.tree.get_children():
            tags = self.tree.item(item, "tags")
            col_name = self.tree.item(item, "values")[0]
            if "selected" in tags:
                selected_columns.append(col_name)

        # Determine discarded columns based on the selected columns
        discarded_columns = [col for col in self.headers if col not in selected_columns]

        # Preprocess data with the appropriate columns to discard
        self.X, _ = preprocess_data(self.dataset,
                                    discard_columns=discarded_columns)
        self.dataset = self.dataset.drop(columns=discarded_columns)

        logging.info(
            f"Preprocessing completed for dataset: {self.dataset_name}. Shape after preprocessing: {self.dataset}")

        self.causal_graph_pre = causal_discovery(self.X, 'pc', num_jobs=0).dpath.T
        self.causal_graph = postprocess_cd(self.causal_graph_pre, allow_self_loops=False, interpolate=True)

        logging.info(f"Causal discovery completed for dataset: {self.dataset_name}. Graph path: {self.causal_graph}")

        # Fetch hyperparameters from the dictionary
        hp = self.hyperparams
        self.shape = self.dataset.shape
        self.vae = create_vae_model(
            self.shape,
            self.causal_graph
        )
        logging.info(f"VAE model created for dataset: {self.dataset_name}")

        self.discriminator = create_gan_discriminator(
            shape=self.shape
        )
        logging.info(f"Discriminator model created.")

        self.gan = create_gan_model(
            self.vae,
            self.discriminator
        )
        logging.info(
            f"GAN model created for dataset: {self.dataset_name} with epochs={hp['epochs']} and batch_size_factor={hp['batch_size_factor']}")

        # Enable the Train and Analysis buttons
        self.button_train.config(state=NORMAL)
        self.button_run_analysis.config(state=DISABLED)  # Analysis is only enabled after training

        self.button_plot_errors.config(state=DISABLED)
        self.button_plot_causal_graph.config(state=NORMAL)
        self.button_plot_all_models.config(state=NORMAL)

        self.update_history(dataset_name=self.dataset_name, processes={"Data Preprocessing": {}},
                            details={"Encoding": "Label",
                                     "Normalization": "z-score"})
        messagebox.showinfo("Dataset Processed",
                            f"Processed dataset: {self.dataset_name}. Model and Causal Info available. You may also "
                            f"train the model.")

    def train_models(self):
        seed(0)
        tf.random.set_seed(0)

        # Ensure models are created
        if not all([self.vae, self.gan, self.discriminator]):
            messagebox.showerror("Error", "Models are not properly created. Please process the dataset first.")
            return

        # Create a progress bar widget
        self.progress_bar = ttk.Progressbar(self.root, orient=HORIZONTAL, length=400, mode='determinate')
        self.progress_bar.pack(pady=20)

        self.progress_label_var = StringVar()
        self.progress_label_var.set(f"Training model on dataset {self.dataset_name}... | 0%")
        self.progress_label = ttk.Label(root, textvariable=self.progress_label_var)
        self.progress_label.pack(pady=10)

        # Fetch hyperparameters from the dictionary
        hp = self.hyperparams

        # Create queues to hold results and progress updates
        self.result_queue = Queue()
        self.progress_queue = Queue()

        # Start the thread for training the models
        self.training_thread = threading.Thread(
            target=self._train_models_in_thread,
            args=(
                self.X, self.vae, self.gan, self.discriminator, hp['epochs'],
                self.X.shape[0] // hp['batch_size_factor'],
                self.result_queue, self.progress_queue
            )
        )
        self.training_thread.start()

        # Periodically check if the thread has completed and update progress
        self.root.after(100, self.check_training_thread)

    def check_training_thread(self):
        # Check if the thread is still running
        if self.training_thread.is_alive():
            # Update progress bar
            if not self.progress_queue.empty():
                progress = self.progress_queue.get()
                self.progress_bar['value'] = progress
                self.progress_bar.update_idletasks()
                self.progress_label_var.set((f"Training model on dataset {self.dataset_name}... | {progress:.2f}%"))

            # Check again after 100ms
            self.root.after(10, self.check_training_thread)
        else:
            # If the thread has finished, retrieve the results
            self.reconstruction_errors = self.result_queue.get()
            self.d_losses = self.result_queue.get()
            self.gan_losses = self.result_queue.get()

            # Log completion
            logging.info("Models trained successfully.")

            # Update history
            self.update_history(
                dataset_name=self.dataset_name,
                train_sessions={"Model Training": {}},
                details={"Reconstruction errors": list(self.reconstruction_errors),
                         "GAN loss": list(self.gan_losses)[-1],
                         "Discriminator loss": list(self.d_losses)[-1]}
            )

            # Notify the user
            messagebox.showinfo(
                "Model Trained",
                f"Trained model with dataset: {self.dataset_name}. Reconstruction errors available. You may now analyse the results via ensemble Isolation Forests."
            )

            # Enable the Analysis button
            self.button_run_analysis.config(state=NORMAL)
            self.button_plot_errors.config(state=NORMAL)
            self.button_plot_causal_graph.config(state=NORMAL)
            self.button_plot_all_models.config(state=NORMAL)

            # Destroy the progress bar once training is complete
            self.progress_bar.destroy()
            self.progress_label.destroy()

    @classmethod
    def _train_models_in_thread(cls, X, vae, gan, discriminator, epochs, batch_size, result_queue, progress_queue):
        try:
            logging.info("Starting training process...")

            # Train models with progress
            d_losses, gan_losses = train_models(
                X, vae, gan, discriminator,
                epochs=epochs,
                batch_size=batch_size,
                progress_queue=progress_queue
            )

            logging.info("Training completed. Evaluating model...")

            # Evaluate model
            reconstruction_errors = evaluate_model(X, vae, plot=False)

            logging.info(f"Evaluation completed. Errors: {reconstruction_errors}")

            # Put results in the queue
            result_queue.put(reconstruction_errors)
            result_queue.put(d_losses)
            result_queue.put(gan_losses)
        except Exception as e:
            logging.error(f"An error occurred in the training process: {e}")
            result_queue.put(None)  # Or handle the error as needed

    def open_advanced_dialog(self):
        top = Toplevel(self.root)
        top.title("Advanced Configurations")

        # Dropdown for causal algorithm selection
        Label(top, text="Select Causal Algorithm:").grid(row=0, column=0, sticky=W, padx=5)

        # Variable for holding selected algorithm
        algorithm_var = StringVar(top)
        algorithm_var.set(self.hyperparams.get('causal_algorithm', 'ges'))  # Default value

        # OptionMenu for selecting algorithm
        algorithm_menu = OptionMenu(top, algorithm_var, 'ges', 'pc')
        algorithm_menu.grid(row=0, column=1, sticky=W)


        Label(top, text="Isolation Forest Contamination Start:").grid(row=6, column=0, sticky=W, padx=5)
        contamination_start_entry = Entry(top, width=10)
        contamination_start_entry.insert(END, str(self.hyperparams['contamination_start']))
        contamination_start_entry.grid(row=6, column=1, sticky=W)

        Label(top, text="Isolation Forest Contamination End:").grid(row=7, column=0, sticky=W, padx=5)
        contamination_end_entry = Entry(top, width=10)
        contamination_end_entry.insert(END, str(self.hyperparams['contamination_end']))
        contamination_end_entry.grid(row=7, column=1, sticky=W)

        Label(top, text="Isolation Forest Contamination Step:").grid(row=8, column=0, sticky=W, padx=5)
        contamination_step_entry = Entry(top, width=10)
        contamination_step_entry.insert(END, str(self.hyperparams['contamination_step']))
        contamination_step_entry.grid(row=8, column=1, sticky=W)

        Label(top, text="Isolation Forest Number of Trials:").grid(row=9, column=0, sticky=W, padx=5)
        num_trials_entry = Entry(top, width=10)
        num_trials_entry.insert(END, str(self.hyperparams['num_trials']))
        num_trials_entry.grid(row=9, column=1, sticky=W)

        Label(top, text="Epochs:").grid(row=10, column=0, sticky=W, padx=5)
        epochs_entry = Entry(top, width=10)
        epochs_entry.insert(END, str(self.hyperparams['epochs']))
        epochs_entry.grid(row=10, column=1, sticky=W)

        Label(top, text="Batch Size Factor:").grid(row=11, column=0, sticky=W, padx=5)
        batch_size_entry = Entry(top, width=10)
        batch_size_entry.insert(END, str(self.hyperparams['batch_size_factor']))
        batch_size_entry.grid(row=11, column=1, sticky=W)

        def apply_changes():
            self.hyperparams['causal_algorithm'] = algorithm_var.get()
            self.hyperparams['contamination_start'] = float(contamination_start_entry.get())
            self.hyperparams['contamination_end'] = float(contamination_end_entry.get())
            self.hyperparams['contamination_step'] = float(contamination_step_entry.get())
            self.hyperparams['num_trials'] = int(num_trials_entry.get())
            self.hyperparams['epochs'] = int(epochs_entry.get())
            self.hyperparams['batch_size_factor'] = int(batch_size_entry.get())
            top.destroy()

        Button(top, text="Apply", command=apply_changes).grid(row=13, column=0, columnspan=2, pady=10)

    def run_analysis(self):
        seed(0)
        tf.random.set_seed(0)
        # Create a queue to hold the results from the process
        self.result_queue = multiprocessing.Queue()
        self.progress_queue = multiprocessing.Queue()

        # Create a progress bar widget
        self.progress_bar = ttk.Progressbar(self.root, orient=HORIZONTAL, length=400, mode='determinate')
        self.progress_bar.pack(pady=20)

        self.progress_label_var = StringVar()
        self.progress_label_var.set((f"Performing EIF analysis on dataset {self.dataset_name}... | 0%"))
        self.progress_label = ttk.Label(root, textvariable=self.progress_label_var)
        self.progress_label.pack(pady=10)

        # Start the process for running the Isolation Forest analysis
        self.process = multiprocessing.Process(
            target=self._train_isolation_forest_in_process,
            args=(self.dataset, self.dataset.columns, self.reconstruction_errors,
                  self.hyperparams, self.result_queue, self.progress_queue)
        )
        self.process.start()

        # Periodically check if the process has completed
        self.root.after(100, self.check_analysis_process)

    def check_analysis_process(self):
        # Check if the process is still running
        if self.process.is_alive():
            # If the process is still running, check again after 100ms
            self.root.after(10, self.check_analysis_process)

            # Update progress bar
            if not self.progress_queue.empty():
                progress = float(self.progress_queue.get())
                self.progress_bar['value'] += progress
                self.progress_bar.update_idletasks()
                self.progress_label_var.set((f"Performing EIF analysis on dataset {self.dataset_name}... | {self.progress_bar['value']:.2f}%"))
        else:
            # If the process has finished, retrieve the results
            quasi_identifiers_indices, quasi_identifiers, weighted_votes, results = self.result_queue.get()

            # Assign the results to the class variables
            self.quasi_identifiers_indices = list(quasi_identifiers_indices)
            self.quasi_identifiers = list(map(lambda x: x.strip(), quasi_identifiers))
            self.weighted_votes = weighted_votes
            self.isolation_results = results

            # Log completion
            logging.info("Isolation Forest analysis completed.")

            # Update history
            self.update_history(
                dataset_name=self.dataset_name,
                analysis_results={"Ensemble Isolation Forests Analysis": {}},
                details={"QIDs": list(self.quasi_identifiers), "weights": self.weighted_votes}
            )

            # Notify the user
            messagebox.showinfo(
                "EIF Analysis completed",
                f"Ensemble Isolation Forest Analysis completed for dataset: {self.dataset_name}. Report available."
            )

            # Destroy the progress bar once training is complete
            self.progress_bar.destroy()
            self.progress_label.destroy()

            # Enable the report button after the analysis is complete
            self.button_view_report.config(state=NORMAL)

            if self.ground_truth is not None:
                self.button_compare_qid.config(state=NORMAL)

    @classmethod
    def _train_isolation_forest_in_process(cls, dataset, columns, reconstruction_errors, hyperparams, result_queue, progress_queue):
        # Train the Isolation Forest ensemble in a separate process
        quasi_identifiers_indices, quasi_identifiers, weighted_votes, results = train_isolation_forest_ensemble(
            dataset, columns, reconstruction_errors,
            contamination_start=hyperparams['contamination_start'],
            contamination_end=hyperparams['contamination_end'],
            contamination_step=hyperparams['contamination_step'],
            num_trials=hyperparams['num_trials'],
            progress_queue=progress_queue
        )

        # Put the results in the queue to return them to the main process
        result_queue.put((quasi_identifiers_indices, quasi_identifiers, weighted_votes, results))

    def view_report(self):
        # Generate the report and get the file path
        report_path = generate_explanation_report(self.dataset,
            self.dataset_name, self.dataset.columns, self.quasi_identifiers_indices, self.weighted_votes,
            self.reconstruction_errors, self.causal_graph_pre, self.causal_graph
        )

        if not os.path.exists(report_path):
            logging.error(f"Report not found at {report_path}")
            return

        # Open the HTML file in the default web browser
        webbrowser.open(report_path)

        logging.info("Explanation report opened in the default web browser.")

    def plot_reconstruction_errors(self):
        plot_error_histograms(self.reconstruction_errors, self.dataset.columns)
        # Show the plot
        plt.show()

    def plot_causal_graph(self):
        plot_causal_graph(self.causal_graph_pre, self.causal_graph, self.dataset.columns)

    def plot_all_models(self):
        if self.vae and self.discriminator and self.gan:
            # Plot and save the VAE model architecture
            plot_model(self.vae, to_file='vae_model.png', show_shapes=True, show_layer_names=True)
            vae_img = plt.imread('vae_model.png')

            # Plot and save the Discriminator model architecture
            plot_model(self.discriminator, to_file='discriminator_model.png', show_shapes=True, show_layer_names=True)
            discriminator_img = plt.imread('discriminator_model.png')

            # Plot and save the GAN model architecture
            plot_model(self.gan, to_file='gan_model.png', show_shapes=True, show_layer_names=True)
            gan_img = plt.imread('gan_model.png')

            # Create a fullscreen figure with three equally divided columns
            plt.figure(figsize=(18, 6))  # Adjust figure size for better clarity

            # Plot the VAE model architecture
            plt.subplot(1, 3, 1)
            plt.imshow(vae_img)
            plt.axis('off')
            plt.title('VAE Model Architecture')

            # Plot the Discriminator model architecture
            plt.subplot(1, 3, 2)
            plt.imshow(discriminator_img)
            plt.axis('off')
            plt.title('Discriminator Model Architecture')

            # Plot the GAN model architecture
            plt.subplot(1, 3, 3)
            plt.imshow(gan_img)
            plt.axis('off')
            plt.title('GAN Model Architecture')

            # Display the plots
            plt.show()


if __name__ == "__main__":
    java_env = 'C:/Program Files/Java/jdk-22/bin'
    os.environ["PATH"] += os.pathsep + os.pathsep.join([java_env])
    multiprocessing.set_start_method('spawn')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    matplotlib.use('TkAgg')
    root = Tk()
    app = QuasiIdentifierApp(root)
    root.mainloop()
