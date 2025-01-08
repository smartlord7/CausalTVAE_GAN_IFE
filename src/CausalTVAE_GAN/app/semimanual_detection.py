import copy
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import matplotlib.colors as mcolors


class MetricInspectionTool:
    colors = ["red", "yellow", "green"]
    n_bins = 100  # Number of discrete bins for the colormap
    cmap_name = "green_yellow_red"
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    def __init__(self, root):
        self.root = root
        self.root.title("Quasi-Identifier Inspection Tool")
        self.step = 0
        self.history = []
        self.current_selection = []
        self.undo_stack = []
        self.redo_stack = []

        style = ttk.Style()
        style.configure("Treeview", font=("Helvetica", 12))  # Increase font size

        # Data initialization
        self.attributes = [f"Attribute {i + 1}" for i in range(10)]
        self.metrics = ["Distinction", "Separation"]
        self.data = self.generate_random_data()

        # Frame for graphs
        self.graph_frame = ttk.Frame(root)
        self.graph_frame.pack(fill=tk.BOTH, expand=True)

        # Central frame for table and current state
        self.data_frame = ttk.Frame(root)
        self.data_frame.pack(fill=tk.BOTH, expand=True)

        # Frame for history and suggestions
        self.history_frame = ttk.Frame(root)
        self.history_frame.pack(expand=True)

        # Initialize UI
        self.create_graph()
        self.create_data_tables()
        self.create_suggestions()
        self.create_current_state()
        self.create_history()
        self.create_controls()

    def generate_random_data(self):
        np.random.seed(0)
        num_attrs = len(self.attributes)
        data = {
            'attributes': self.attributes,
            'metrics': {metric: np.random.rand(num_attrs) for metric in self.metrics},
            'deltas': {metric: np.random.rand(num_attrs) for metric in self.metrics}
        }
        return data

    def create_data_tables(self):
        # Define columns
        columns = ["Attribute"] + self.metrics + [f"Delta {metric}" for metric in self.metrics]

        # Frame for treeviews
        self.treeviews_frame = ttk.Frame(self.data_frame)
        self.treeviews_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create Treeview widget for attributes
        self.attr_treeview = ttk.Treeview(self.treeviews_frame, columns=["Attribute"], show='headings',
                                          selectmode="extended")
        self.attr_treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.attr_treeview.heading("Attribute", text="Attribute")
        self.attr_treeview.column("Attribute", width=100, anchor=tk.CENTER)

        # Create separate Treeview widgets for each metric and delta
        self.treeviews = {}
        for metric in self.metrics:
            tv = ttk.Treeview(self.treeviews_frame, columns=[metric], show='headings', selectmode="extended")
            tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            tv.heading(metric, text=metric)
            tv.column(metric, width=100, anchor=tk.CENTER)
            self.treeviews[metric] = tv

        for metric in self.metrics:
            delta_metric = f"Delta {metric}"
            tv = ttk.Treeview(self.treeviews_frame, columns=[delta_metric], show='headings', selectmode="extended")
            tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            tv.heading(delta_metric, text=delta_metric)
            tv.column(delta_metric, width=100, anchor=tk.CENTER)
            self.treeviews[delta_metric] = tv

        # Add Select Button
        self.select_button = tk.Button(self.data_frame, text="Select Attributes", command=self.select_attributes)
        self.select_button.pack(side=tk.BOTTOM, pady=5)

        # Populate the tables with data
        self.populate_data_tables()

    def populate_data_tables(self):
        # Clear existing data in all treeviews
        self.attr_treeview.delete(*self.attr_treeview.get_children())
        for tv in self.treeviews.values():
            tv.delete(*tv.get_children())

        # Populate the attribute treeview and metric/delta treeviews
        for i, attr in enumerate(self.attributes):
            if attr in self.current_selection:
                continue  # Skip already selected attributes
            attr_values = [attr]
            metrics_values = [f"{self.data['metrics'][metric][i]:.2f}" for metric in self.metrics]
            delta_values = [f"{self.data['deltas'][metric][i]:.2f}" for metric in self.metrics]

            # Insert rows into attribute treeview
            self.attr_treeview.insert("", "end", values=attr_values)

            # Insert rows into metric treeviews
            for metric, value in zip(self.metrics, metrics_values):
                self.treeviews[metric].insert("", "end", values=[value])

            # Insert rows into delta treeviews
            for metric, value in zip(self.metrics, delta_values):
                delta_metric = f"Delta {metric}"
                self.treeviews[delta_metric].insert("", "end", values=[value])

        # Apply color to cells
        self.color_tables()
        self.generate_suggestions()

    def color_tables(self):
        # Dictionary to hold min and max values for each metric column
        column_min_max = {metric: (float('inf'), float('-inf')) for metric in self.metrics}
        delta_column_min_max = {f"Delta {metric}": (float('inf'), float('-inf')) for metric in self.metrics}
        # Define the custom colormap
        colors = ["red", "yellow", "green"]
        custom_cmap = mcolors.LinearSegmentedColormap.from_list("green_yellow_red", colors, N=100)

        # First pass: Determine min and max for each metric and delta column
        for metric in self.metrics:
            for item in self.treeviews[metric].get_children():
                values = self.treeviews[metric].item(item, 'values')
                try:
                    value = float(values[0])
                    min_val, max_val = column_min_max[metric]
                    column_min_max[metric] = (min(min_val, value), max(max_val, value))
                except ValueError:
                    continue

        for metric in self.metrics:
            for item in self.treeviews[f"Delta {metric}"].get_children():
                values = self.treeviews[f"Delta {metric}"].item(item, 'values')
                try:
                    value = float(values[0])
                    min_val, max_val = delta_column_min_max[f"Delta {metric}"]
                    delta_column_min_max[f"Delta {metric}"] = (min(min_val, value), max(max_val, value))
                except ValueError:
                    continue

        # Apply colors based on the normalized values
        for metric in self.metrics:
            for item in self.treeviews[metric].get_children():
                values = self.treeviews[metric].item(item, 'values')
                try:
                    value = float(values[0])
                    min_val, max_val = column_min_max[metric]

                    # Normalize the value
                    norm_value = (value - min_val) / (max_val - min_val) if min_val != max_val else 0.5
                    color = custom_cmap(norm_value)
                    color_hex = mcolors.to_hex(color)

                    # Apply color to the cell
                    tag = f"cell_{item}_{metric}"
                    self.treeviews[metric].tag_configure(tag, background=color_hex)
                    self.treeviews[metric].item(item, tags=[tag])
                except ValueError:
                    continue

        for metric in self.metrics:
            for item in self.treeviews[f"Delta {metric}"].get_children():
                values = self.treeviews[f"Delta {metric}"].item(item, 'values')
                try:
                    value = float(values[0])
                    min_val, max_val = delta_column_min_max[f"Delta {metric}"]

                    # Normalize the value
                    norm_value = (value - min_val) / (max_val - min_val) if min_val != max_val else 0.5
                    color = custom_cmap(norm_value)
                    color_hex = mcolors.to_hex(color)

                    # Apply color to the cell
                    tag = f"cell_{item}_{metric}"
                    self.treeviews[f"Delta {metric}"].tag_configure(tag, background=color_hex)
                    self.treeviews[f"Delta {metric}"].item(item, tags=[tag])
                except ValueError:
                    continue

    def normalize(self, value, min_val, max_val):
        if min_val == max_val:
            return 0.5  # Avoid division by zero; midpoint color
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
        return norm(value)

    def get_color(self, norm_value):
        color = plt.cm.viridis(norm_value)
        return mcolors.to_hex(color)


    def create_suggestions(self):
        self.suggestions_label = tk.Label(self.data_frame, text="Suggestions")
        self.suggestions_label.pack(anchor=tk.W, pady=5)

        self.suggestions_listbox = tk.Listbox(self.data_frame, selectmode=tk.MULTIPLE, height=15)
        self.suggestions_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.apply_suggestions_button = tk.Button(self.data_frame, text="Apply Suggestions",
                                                  command=self.apply_suggestions)
        self.apply_suggestions_button.pack(pady=5)

    def create_graph(self):
        self.fig, self.axs = plt.subplots(nrows=len(self.metrics), ncols=2, figsize=(12, len(self.metrics) * 2))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize empty plots
        for i, metric in enumerate(self.metrics):
            self.axs[i, 0].set_title(f"{metric}")
            self.axs[i, 1].set_title(f"Delta {metric}")
            self.axs[i, 0].set_xlabel("Step")
            self.axs[i, 0].set_ylabel(metric)
            self.axs[i, 1].set_xlabel("Step")
            self.axs[i, 1].set_ylabel(f"Delta {metric}")

        self.fig.tight_layout()
        self.canvas.draw()

    def update_graph(self):
        for i, metric in enumerate(self.metrics):
            steps = list(range(len(self.current_selection)))
            metrics_values = [self.data['metrics'][metric][self.attributes.index(attr)] for attr in
                              self.current_selection]
            deltas_values = [self.data['deltas'][metric][self.attributes.index(attr)] for attr in
                             self.current_selection]

            self.axs[i, 0].clear()
            self.axs[i, 1].clear()

            self.axs[i, 0].plot(steps, metrics_values, label=metric, color='blue', marker='o')
            self.axs[i, 1].plot(steps, deltas_values, label=f'Delta {metric}', color='red', marker='o')

            self.axs[i, 0].set_title(f"{metric}")
            self.axs[i, 0].legend()
            self.axs[i, 1].set_title(f"Delta {metric}")
            self.axs[i, 1].legend()

        self.fig.tight_layout()
        self.canvas.draw()

    def create_current_state(self):
        self.state_frame = ttk.Frame(self.data_frame)
        self.state_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.state_label = tk.Label(self.state_frame, text="Current State")
        self.state_label.pack(anchor=tk.W, padx=5, pady=5)

        self.state_text = tk.Text(self.state_frame, height=10, wrap=tk.WORD)
        self.state_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.update_current_state()

    def update_current_state(self):
        self.state_text.delete(1.0, tk.END)
        if self.current_selection:
            self.state_text.insert(tk.END, f"Iteration: {self.step}\n\n")
            self.state_text.insert(tk.END, "Selected Attributes:\n")
            for attr in self.current_selection:
                self.state_text.insert(tk.END, f"  {attr}\n")

            self.state_text.insert(tk.END, "\nMetrics:\n")
            for metric in self.metrics:
                values = [self.data['metrics'][metric][self.attributes.index(attr)] for attr in self.current_selection]
                self.state_text.insert(tk.END, f"{metric}:\n")
                for attr, value in zip(self.current_selection, values):
                    self.state_text.insert(tk.END, f"  {attr}: {value:.2f}\n")

            # Add suggestions
            suggestions = self.generate_suggestions()
            self.suggestions_listbox.delete(0, tk.END)  # Clear existing suggestions
            for suggestion in suggestions:
                self.suggestions_listbox.insert(tk.END, suggestion)

            self.state_text.insert(tk.END, "Suggestions:\n")
            self.state_text.insert(tk.END, "\n".join(suggestions))

        else:
            self.state_text.insert(tk.END, "No attributes selected.\n")

    def create_history(self):
        self.history_text = tk.Text(self.history_frame, height=10, wrap=tk.WORD)
        self.history_text.pack(fill=tk.BOTH, expand=True)
        self.update_history()

    def update_history(self):
        self.history_text.delete(1.0, tk.END)
        for entry in self.history:
            self.history_text.insert(tk.END, entry + "\n")

    def create_controls(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.BOTH, expand=True)

        # Rollback and Forward buttons
        self.rollback_button = tk.Button(control_frame, text="Rollback", command=self.rollback)
        self.rollback_button.pack(side=tk.LEFT, padx=5)
        self.rollback_button.config(state=tk.DISABLED)

        self.forward_button = tk.Button(control_frame, text="Forward", command=self.forward)
        self.forward_button.pack(side=tk.LEFT, padx=5)
        self.forward_button.config(state=tk.DISABLED)

        self.apply_suggestions_button.pack(side=tk.LEFT, padx=5)
        self.apply_suggestions_button.config(state=tk.DISABLED)

    def select_attributes(self):
        selected_items = self.attr_treeview.selection()
        if not selected_items:
            messagebox.showwarning("Selection Error", "Please select one or more attributes from the table.")
            return

        # Get the selected attributes
        selected_attrs = [self.attr_treeview.item(item, 'values')[0] for item in selected_items]

        # Filter out already selected attributes
        new_attrs = [attr for attr in selected_attrs if attr not in self.current_selection]

        if not new_attrs:
            messagebox.showinfo("Info", "All selected attributes are already in the current selection.")
            return

        # Update current selection
        self.current_selection.extend(new_attrs)
        self.step += 1
        self.add_to_history(f"Selected attributes: {', '.join(new_attrs)}")
        self.undo_stack.append({
            'data': copy.deepcopy(self.data),
            'current_selection': copy.deepcopy(self.current_selection)
        })
        self.redo_stack.clear()

        # Update UI components
        self.update_data()
        self.update_graph()
        self.update_current_state()
        self.color_tables()

        # Enable/Disable buttons
        self.rollback_button.config(state=tk.NORMAL)
        self.apply_suggestions_button.config(state=tk.NORMAL)

    def update_data(self):
        # Exclude already selected attributes from the update
        num_attrs = len(self.attributes)
        selected_indices = [self.attributes.index(attr) for attr in self.current_selection if attr in self.attributes]

        for metric in self.metrics:
            self.data['metrics'][metric] = np.random.rand(num_attrs)
            self.data['deltas'][metric] = np.random.rand(num_attrs)

        # Re-populate the data table
        self.populate_data_tables()

    def add_to_history(self, action):
        self.history.append(f"Step {self.step}: {action}")
        self.update_history()

    def rollback(self):
        if not self.undo_stack:
            messagebox.showinfo("Info", "No more steps to rollback.")
            return

        # Save the current state (data and current_selection) to redo stack
        self.redo_stack.append({
            'data': copy.deepcopy(self.data),
            'current_selection': copy.deepcopy(self.current_selection)
        })

        # Restore the previous state (data and current_selection) from undo stack
        state = self.undo_stack.pop()
        self.data = state['data']
        self.current_selection = state['current_selection']

        # Decrement the step counter
        self.step -= 1

        # Ensure that the data update reflects the rolled-back state
        self.update_data()

        # Update UI components
        self.update_graph()
        self.update_current_state()
        self.color_tables()

        # Log the action
        self.add_to_history("Rollback")

        # Enable/Disable buttons based on the new state of the stacks
        if not self.undo_stack:
            self.rollback_button.config(state=tk.DISABLED)
        self.forward_button.config(state=tk.NORMAL)

        # Ensure redo stack is not empty to allow redo operations
        if not self.redo_stack:
            self.forward_button.config(state=tk.DISABLED)

    def forward(self):
        if not self.redo_stack:
            messagebox.showinfo("Info", "No more steps to forward.")
            return

        # Save current state (data and current_selection) to undo stack
        self.undo_stack.append({
            'data': copy.deepcopy(self.data),
            'current_selection': copy.deepcopy(self.current_selection)
        })

        # Restore the next state (data and current_selection) from redo stack
        state = self.redo_stack.pop()
        self.data = state['data']
        self.current_selection = state['current_selection']

        # Increment the step counter
        self.step += 1

        # Update UI components
        self.update_data()
        self.update_graph()
        self.update_current_state()
        self.color_tables()

        # Log the action
        self.add_to_history("Forward")

        # Enable/Disable buttons based on the new state of the stacks
        if not self.redo_stack:
            self.forward_button.config(state=tk.DISABLED)
        self.rollback_button.config(state=tk.NORMAL)

    def generate_suggestions(self):
        suggestions = []

        # Calculate current average metrics and deltas for the selected attributes
        current_metric_avg = {metric: np.mean([self.data['metrics'][metric][self.attributes.index(attr)]
                                               for attr in self.current_selection])
                              for metric in self.metrics}
        current_delta_avg = {metric: np.mean([self.data['deltas'][metric][self.attributes.index(attr)]
                                              for attr in self.current_selection])
                             for metric in self.metrics}

        # Identify attributes with significant delta or metric values
        for metric in self.metrics:
            metric_values = [self.data['metrics'][metric][self.attributes.index(attr)] for attr in self.attributes]
            delta_values = [self.data['deltas'][metric][self.attributes.index(attr)] for attr in self.attributes]

            # Calculate improvement potential for each attribute
            for attr, metric_value, delta in zip(self.attributes, metric_values, delta_values):
                if attr not in self.current_selection and delta > 0.5:  # Example threshold
                    # Calculate potential improvement
                    expected_metric = metric_value
                    improvement = expected_metric - current_metric_avg[metric]

                    suggestions.append(
                        f"Adding '{attr}' could improve '{metric}' by {improvement:.2f}. "
                        f"Current average: {current_metric_avg[metric]:.2f}, Potential new average: {expected_metric:.2f}."
                    )

        return suggestions

    def apply_suggestions(self):
        selected_indices = self.suggestions_listbox.curselection()
        selected_suggestions = [self.suggestions_listbox.get(i) for i in selected_indices]

        if not selected_suggestions:
            messagebox.showwarning("No Selection", "No suggestions selected.")
            return

        # Save current state before making changes
        self.undo_stack.append({
            'data': copy.deepcopy(self.data),
            'current_selection': copy.deepcopy(self.current_selection)
        })
        self.redo_stack.clear()

        for suggestion in selected_suggestions:
            attr = suggestion.split("'")[1]  # Extract attribute name from suggestion
            if attr not in self.current_selection:
                self.current_selection.append(attr)

        self.step += 1
        self.add_to_history(f"Applied suggestions: {', '.join(selected_suggestions)}")

        self.update_data()
        self.update_graph()
        self.update_current_state()
        self.color_tables()

        # Enable/Disable buttons
        self.rollback_button.config(state=tk.NORMAL)
        self.apply_suggestions_button.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = MetricInspectionTool(root)
    root.mainloop()
