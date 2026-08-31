import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Preprocessing: add an 'Algorithm' column that splits SVM into its
#    linear / rbf kernel variants. Run once per CSV (it overwrites the file).
# ---------------------------------------------------------------------------
def preprocess_metric_values(filename):
    df = pd.read_csv(filename)
    df['Algorithm'] = np.where(
        df['Model'] == 'SVM',
        np.where(df['Kernel'] == 'rbf', 'SVM-R', 'SVM-L'),
        df['Model']
    )
    df.to_csv(filename, index=False)


# ---------------------------------------------------------------------------
# 2. Stats: for each requested question / algorithm, compute the overall
#    mean, std, and sem of Test_Metric (pooled across N_Features / chip
#    combos - this is the "Redo" style aggregation, one number per bar).
# ---------------------------------------------------------------------------
def compute_question_algorithm_stats(filename, questions, group_names,
                                      algorithm_map=None, expected_n=12):
    """
    Parameters
    ----------
    filename : path to the CSV (must already have an 'Algorithm' column,
               see preprocess_metric_values)
    questions : list of question codes to pull from the CSV, e.g.
                ["Detection", "Type_NoDI", ...]
    group_names : display labels for the bars in each group, e.g.
                  ['LR', 'LDA', 'SVM-L', 'SVM-R']
    algorithm_map : optional dict mapping a group_name -> the actual value
                    in the CSV's 'Algorithm' column (defaults to identity,
                    but e.g. 'SVM-L' -> 'SVM_L' is common since
                    preprocess_metric_values uses underscores)
    expected_n : sanity-check count per (question, algorithm) cell

    Returns
    -------
    means, sems : dicts keyed by question, each value a list aligned with
                  group_names, e.g. means["Detection"] = [m_LR, m_LDA, ...]
    """
    if algorithm_map is None:
        algorithm_map = {name: name for name in group_names}

    df = pd.read_csv(filename)
    df = df.sort_values(by=['Algorithm', 'Question'])

    means, sems = {}, {}

    for q in questions:
        df_q = df[df['Question'] == q]
        means[q] = []
        sems[q] = []
        for name in group_names:
            algo_value = algorithm_map[name]
            df_qa = df_q[df_q['Algorithm'] == algo_value]
            n = df_qa['Test_Metric'].count()
            if n != expected_n:
                print(f"Warning: {q} {name} has n={n}, expected {expected_n}")
            mean = df_qa['Test_Metric'].mean() * 100
            std = df_qa['Test_Metric'].std() * 100
            sem = std / np.sqrt(n) if n > 0 else np.nan
            means[q].append(mean)
            sems[q].append(sem)
            print(
                f"Average test metric value for {q} {name} is "
                f"{mean:.4f} +/- {sem:.4f} (std={std:.4f}, n={n})"
            )

    return means, sems


# ---------------------------------------------------------------------------
# 3. Reshape into the {title: [values...]} dicts the chart function expects,
#    swapping question codes for their display titles and preserving order.
# ---------------------------------------------------------------------------
def build_chart_dicts(means, sems, question_titles, questions):
    data_dict, err_dict = {}, {}
    for q in questions:
        title = question_titles.get(q, q)
        data_dict[title] = means[q]
        err_dict[title] = sems[q]
    return data_dict, err_dict


# ---------------------------------------------------------------------------
# 4. Chart: grouped bars, sem error bars, bold annotation on the tallest
#    bar in each group.
# ---------------------------------------------------------------------------
def create_grouped_bar_chart(data_dict, labels, err_dict=None, colors=None,
                              title='Performance Metrics Comparison',
                              ylabel='Performance Score', figsize=(14, 6)):
    metrics = list(data_dict.keys())
    n_metrics = len(metrics)
    n_sets = len(labels)

    x = np.arange(n_metrics)
    width = 0.8 / n_sets

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_ylim(50, 100)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(axis='y', alpha=0.5, linestyle='-')

    bars_by_metric = {metric: [] for metric in metrics}

    for i, set_name in enumerate(labels):
        offset = width * i - (width * (n_sets - 1) / 2)
        values = [data_dict[metric][i] for metric in metrics]
        errors = [err_dict[metric][i] for metric in metrics] if err_dict else None
        color = colors[i] if colors and i < len(colors) else None

        bars = ax.bar(
            x + offset, values, width,
            yerr=errors, capsize=4,
            error_kw={'elinewidth': 1.2, 'ecolor': '#333333'},
            label=set_name, color=color
        )
        for j, metric in enumerate(metrics):
            err_j = errors[j] if errors else 0
            bars_by_metric[metric].append((bars[j], values[j], err_j, i))

    # Annotate only the top-performing bar in each metric group, in bold,
    # positioned above its error bar so it never overlaps the whisker.
    for j, metric in enumerate(metrics):
        metric_bars = bars_by_metric[metric]
        top_bar, top_value, top_err, top_idx = max(metric_bars, key=lambda b: b[1])
        bar_x = top_bar.get_x() + top_bar.get_width() / 2
        bar_top = top_bar.get_height() + (top_err if top_err else 0)

        ax.annotate(
            f'{top_value:.1f}%',
            xy=(bar_x, bar_top),
            xytext=(bar_x, bar_top + 1.5),
            ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#1a1a1a',
        )

    ax.set_xlabel('Classifier\n(Metric)', fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, ha='center', fontsize=14)
    ax.legend(loc='best')

    plt.tight_layout()
    return fig, ax

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def _parse_features(feature_str):
    """
    Parse a string like "['BH', 'CH', 'YH', ...]" into an actual Python list,
    without using ast.literal_eval.
    """
    if pd.isna(feature_str):
        return []

    cleaned = feature_str.strip()

    # Strip surrounding brackets if present
    if cleaned.startswith('['):
        cleaned = cleaned[1:]
    if cleaned.endswith(']'):
        cleaned = cleaned[:-1]

    if not cleaned.strip():
        return []

    # Split on commas, then strip whitespace and quote characters from each item
    items = []
    for item in cleaned.split(','):
        item = item.strip()
        # Remove matching single or double quotes
        if len(item) >= 2 and item[0] == item[-1] and item[0] in ("'", '"'):
            item = item[1:-1]
        item = item.strip()
        if item:
            items.append(item)

    return items


def FeaturePresenceChart(filename, questions_to_chart=None, feature_display_name_dict=None, question_display_name_dict=None,
                          feature_order=None):
    """
    Parse and collect features from models, then summarize feature presence per question (classifier).

    feature_order: optional list of feature keys (raw names, as they appear in the
        'Features' column) specifying the row order for the chart. If None, features
        are sorted alphabetically. Any features present in the data but not listed in
        feature_order are appended at the end (alphabetically). Any features listed in
        feature_order but not present in the charted questions' data are ignored (with
        a warning).
    """
    if questions_to_chart is None:
        questions_to_chart = ["Detection", "Type_NoDI", "BW_2", "AS", "CB", "LC"]
    if feature_display_name_dict is None:
        feature_display_name_dict= {"AH": "Aspartic Acid high",
                                    "AL": "Aspartic Acid low",
                                    "BH": "BSA high",
                                    "BL": "BSA low",
                                    "CH": "Cysteine high",
                                    "CL": "Cysteine low",
                                    "DH": "Histidine high",
                                    "DL": "Histidine low",
                                    "GH": "Glycine high",
                                    "GL": "Glycine low",
                                    "KH": "Lysine high",
                                    "KL": "Lysine low",
                                    "N0": "DI water",
                                    "QH": "Glutamine high",
                                    "QL": "Glutamine low",
                                    "YH": "Lysozyme high",
                                    "YL": "Lysozyme low",}
    if question_display_name_dict is None:
        question_display_name_dict= {"Detection": "Presence\n(SVM-R)",
                                     "Type_NoDI": "Type\n(SVM-R)",
                                     "BW": "Presence in\nBottled Water\n(LDA)",
                                     "BW_2": "Presence in\nBottled Water\n(SVM-R)",
                                     "AS": "Presence in\nAmmonium\nSulfate\n(LR)",
                                     "CB": "Presence in\nBlack Carbon\n(SVM-R)",
                                     "LC": "Presence with\nLower\nConcentrations\n(SVM-R)",}

    if feature_order is None:
        feature_order = ['BH', 'BL', 'YH', 'YL', 'KH', 'KL', 'DH', 'DL', 'AH', 'AL', 'QH', 'QL', 'CH', 'CL', 'GH', 'GL', 'N0']
    # ---- Load data ----
    df = pd.read_csv(filename)

    # ---- Collect feature counts per question ----
    # Structure: {question: {feature: count_of_models_using_it}}
    question_feature_counts = {}

    for question, group in df.groupby('Question'):
        n_models = len(group)
        if n_models != 12:
            print(f"Warning: Question '{question}' has {n_models} rows, expected 12.")

        feature_counts = {}
        for feature_str in group['Features']:
            features = _parse_features(feature_str)
            for feat in features:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1

        question_feature_counts[question] = feature_counts

    # ---- Restrict to questions requested for charting ----
    questions_to_chart = [q for q in questions_to_chart if q in question_feature_counts]
    if not questions_to_chart:
        print("No matching questions found to chart.")
        return None

    # ---- Determine the full set of features (rows) across the charted questions ----
    all_features_set = set()
    for q in questions_to_chart:
        all_features_set.update(question_feature_counts[q].keys())

    if feature_order is not None:
        # Warn about requested features that don't actually appear in the data
        missing = [f for f in feature_order if f not in all_features_set]
        if missing:
            print(f"Warning: feature_order includes features not present in the "
                  f"charted questions' data: {missing}")

        # Use the requested order for features that exist, then append any
        # leftover features (present in data but not in feature_order) alphabetically
        ordered = [f for f in feature_order if f in all_features_set]
        leftover = sorted(all_features_set - set(ordered))
        all_features = ordered + leftover
    else:
        all_features = sorted(all_features_set)

    # ---- Build the count matrix: rows = features, columns = questions ----
    matrix = np.zeros((len(all_features), len(questions_to_chart)), dtype=int)
    for col_idx, q in enumerate(questions_to_chart):
        counts = question_feature_counts[q]
        for row_idx, feat in enumerate(all_features):
            matrix[row_idx, col_idx] = counts.get(feat, 0)

    # ---- Labels ----
    feature_labels = [feature_display_name_dict.get(f, f) for f in all_features]
    question_labels = [question_display_name_dict.get(q, q) for q in questions_to_chart]

    # ---- Custom colormap ----
    colors_list = ['#ffffff', '#fde8d0', '#f4b47a', '#d47a3a', '#8c4a1a']
    cmap = LinearSegmentedColormap.from_list('custom_diverging', colors_list)
    cmap.set_bad(color='#cccccc')  # Grey for missing values

    # ---- Plot ----
    fig_width = max(6, 1.5 * len(questions_to_chart) + 2)
    fig_height = max(6, 0.35 * len(all_features) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=12, aspect='auto')

    # Colorbar 0-12
    cbar = fig.colorbar(im, ax=ax, ticks=range(0, 13))
    cbar.set_label('Number of models (of 12) using feature')

    # Axis ticks/labels (major ticks: centered on cells, for labels)
    ax.set_xticks(np.arange(len(question_labels)))
    ax.set_xticklabels(question_labels, ha='center')

    ax.set_yticks(np.arange(len(feature_labels)))
    ax.set_yticklabels(feature_labels)

    # Minor ticks: at cell boundaries, used purely for gridlines
    ax.set_xticks(np.arange(-0.5, len(question_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(feature_labels), 1), minor=True)

    # Solid black grid lines delineating each row/column
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)  # hide minor tick marks themselves

    # Annotate non-zero cells with black numbers
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            val = matrix[row_idx, col_idx]
            if val != 0:
                ax.text(col_idx, row_idx, str(val),
                        ha='center', va='center', color='black')

    ax.set_title("B. Feature Presence Across Classifiers", fontweight="bold", fontsize=18)
    fig.tight_layout()
    plt.show()

    return fig, matrix, all_features, questions_to_chart

# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    filename = 'HPC_results_cm_all.csv'

    # Run once (uncomment) if the CSV doesn't yet have an 'Algorithm' column
    # preprocess_metric_values(filename)

    questions = [
        "Detection", "Type_NoDI", "BW_2",
        "AS", "CB", "LC",
    ]

    group_names = ['LR', 'LDA', 'SVM-L', 'SVM-R']
    # Map display labels to the actual 'Algorithm' values written by
    # preprocess_metric_values (which uses underscores for SVM variants).
    algorithm_map = {'LR': 'LR', 'LDA': 'LDA', 'SVM-L': 'SVM-L', 'SVM-R': 'SVM-R'}
    colors = ['#d1e5f0', '#92c5de', '#4393c3', '#2166ac']

    question_titles = {
        "Detection": "Presence\n(F1 Score)",
        "Type_NoDI": "Type\n(Accuracy)",
        "BW_2": "Presence in\nBottled Water\n(F1 Score)",
        "AS": "Presence in\nAmmonium\nSulfate\n(F1 Score)",
        "CB": "Presence in\nBlack Carbon\n(F1 Score)",
        "LC": "Presence with\nLower\nConcentrations\n(F1 Score)",
    }

    means, sems = compute_question_algorithm_stats(
        filename, questions, group_names, algorithm_map=algorithm_map
    )
    data_dict, err_dict = build_chart_dicts(means, sems, question_titles, questions)

    fig, ax = create_grouped_bar_chart(
        data_dict,
        group_names,
        err_dict=err_dict,
        colors=colors,
        title='A. Machine Learning Performance',
        ylabel='Performance Score (%)',
    )

    plt.savefig('performance_chart.png', dpi=600, bbox_inches='tight')
    plt.show()
    FeaturePresenceChart('HPC_results.csv')
