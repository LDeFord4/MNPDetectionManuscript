"""
Lexi DeFord
Micro- and Nano- Plastic Detection Project
7/30/2026

Script for "Revision 1" comments; to process the multiple data splits
    - averaging out metric values, etc
"""
import pandas as pd
import numpy as np

def PreprocessMetricValues(filename):
    df = pd.read_csv(filename)
    df['Algorithm'] = np.where(
        df['Model'] == 'SVM',
        np.where(df['Kernel'] == 'rbf', 'SVM-R', 'SVM-L'),
        df['Model']
    )
    df.to_csv(filename, index=False)
def ProcessMetricValuesOLD(filename):
    df = pd.read_csv(filename)
    result_dict = {}
    for q in df["Question"].unique():
        print(q)
        df_question = df[df["Question"] == q]
        result_dict[q] = {}
        for a in df_question["Algorithm"].unique():
            # find best avg (avg of each feature selection num)
            df_question_algo = df_question[df_question["Algorithm"] == a]
            fsn_list = []
            for fsn in df_question_algo["N_Features"].unique():
                df_fsn = df_question_algo[df_question_algo["N_Features"] == fsn]
                fsn_mean = df_fsn["Test_Metric"].mean()
                #print(f"{q} {a} with {fsn} features average test metric value is {fsn_mean}")
                fsn_list.append(fsn_mean)
            result_dict[q][a] = fsn_list
            fsn_max = max(fsn_list)
            print(f"Best average test metric value for {q} {a} is {fsn_max} with {fsn_list.index(fsn_max) + 2} features")

    return result_dict

def ProcessMetricValues(filename):
    df = pd.read_csv(filename)
    df = df.sort_values(by=['Algorithm', 'Question'])
    result_dict = {}
    for q in df["Question"].unique():
        print(q)
        df_question = df[df["Question"] == q]
        result_dict[q] = {}
        for a in df_question["Algorithm"].unique():
            # find best avg (avg of each feature selection num)
            df_question_algo = df_question[df_question["Algorithm"] == a]
            fsn_list = []
            fsn_std_list = []
            fsn_sem_list = []
            for fsn in df_question_algo["N_Features"].unique():
                df_fsn = df_question_algo[df_question_algo["N_Features"] == fsn]
                fsn_mean = df_fsn["Test_Metric"].mean()
                fsn_std = df_fsn["Test_Metric"].std()          # standard deviation
                n = df_fsn["Test_Metric"].count()
                if n != 12: #n should be 12 because there are 12 combinations for each test-val chip combo
                    print(f"{n} doesn't equal 12!!!! {a} {fsn}")
                fsn_sem = fsn_std / np.sqrt(n) if n > 0 else np.nan  # standard error of the mean
                #print(f"{q} {a} with {fsn} features average test metric value is {fsn_mean} +/- {fsn_sem}")
                fsn_list.append(fsn_mean)
                fsn_std_list.append(fsn_std)
                fsn_sem_list.append(fsn_sem)

            result_dict[q][a] = {
                "mean": fsn_list,
                "std": fsn_std_list,
                "sem": fsn_sem_list,
            }

            fsn_max = max(fsn_list)
            best_idx = fsn_list.index(fsn_max)
            print(
                f"Best average test metric value for {q} {a} is "
                f"{fsn_max:.4f} +/- {fsn_sem_list[best_idx]:.4f} "
                f"(std={fsn_std_list[best_idx]:.4f}) with {best_idx + 1} features"
            )

    return result_dict
def ProcessMetricValuesRedo(filename):
    df = pd.read_csv(filename)
    df = df.sort_values(by=['Algorithm', 'Question'])
    result_dict = {}
    for q in df["Question"].unique():
        print(q)
        df_question_only = df[df["Question"] == q]
        result_dict[q] = {}
        for a in df_question_only["Algorithm"].unique():
            # find best avg (avg of each feature selection num)
            df_question = df_question_only[df_question_only["Algorithm"] == a]
            fsn_list = []
            fsn_std_list = []
            fsn_sem_list = []
            a_mean = df_question["Test_Metric"].mean()
            std = df_question["Test_Metric"].std()          # standard deviation
            n = df_question["Test_Metric"].count()
            if n != 12: #n should be 12 because there are 12 combinations for each test-val chip combo
                print(f"{n} doesn't equal 12!!!! {a} {q}")
            sem = std / np.sqrt(n) if n > 0 else np.nan  # standard error of the mean
            #print(f"{q} {a} with {fsn} features average test metric value is {fsn_mean} +/- {fsn_sem}")
            fsn_list.append(a_mean)
            fsn_std_list.append(std)
            fsn_sem_list.append(sem)

            result_dict[q][a] = {
                "mean": fsn_list,
                "std": fsn_std_list,
                "sem": fsn_sem_list,
            }

            fsn_max = max(fsn_list)
            best_idx = fsn_list.index(fsn_max)
            print(
                f"Best average test metric value for {q} {a} is "
                f"{fsn_max:.4f} +/- {fsn_sem_list[best_idx]:.4f} "
                f"(std={fsn_std_list[best_idx]:.4f})"
            )

    return result_dict


"""def parse_cm_string(cm_str):
    Parse a confusion matrix string like '[[ 0  8]\n [ 1 11]]' into a numpy array.
    cleaned = cm_str.strip()
    cleaned = re.sub(r'(?<=\d)\s+(?=-?\d)', ', ', cleaned)
    cleaned = re.sub(r'\]\s*\[', '], [', cleaned)
    return np.array(ast.literal_eval(cleaned))


def parse_labels(raw_labels):
    Parse a labels string like "['Null' 'Plastic']" into a list.
    if not isinstance(raw_labels, str):
        return raw_labels
    cleaned = re.sub(r"'\s+'", "', '", raw_labels.strip())
    return ast.literal_eval(cleaned)


def combine_confusion_matrices(csv_path, question_col='Question', cm_col='CM',
                                labels_col='CM_labels',
                                algorithm_col=None, test_metric_col=None):
    
    Parameters
    ----------
    algorithm_col : str, optional
        Name of a CSV column holding the algorithm name (e.g. 'LR', 'RF').
        If given and present in the CSV, the first value per question group
        is stored in results[question]['algorithm'].
    test_metric_col : str, optional
        Name of a CSV column holding a pre-formatted metric string
        (e.g. 'Accuracy = 91.0%'). If given and present, the first value
        per question group is stored in results[question]['test_metric_text'].
    
    df = pd.read_csv(csv_path)
    df['_cm_parsed'] = df[cm_col].apply(parse_cm_string)

    results = {}
    for question, group in df.groupby(question_col):
        matrices = list(group['_cm_parsed'])
        shapes = {m.shape for m in matrices}
        if len(shapes) > 1:
            raise ValueError(f"Question '{question}' has mismatched CM shapes: {shapes}")

        combined = np.sum(matrices, axis=0)

        labels = None
        if labels_col in group.columns:
            labels = parse_labels(group[labels_col].iloc[0])

        algorithm = None
        if algorithm_col and algorithm_col in group.columns:
            algorithm = group[algorithm_col].iloc[0]

        test_metric_text = None
        if test_metric_col and test_metric_col in group.columns:
            test_metric_text = group[test_metric_col].iloc[0]

        results[question] = {
            'matrix': combined,
            'labels': labels,
            'n_rows': len(group),
            'algorithm': algorithm,
            'test_metric_text': test_metric_text,
        }

    return results


def apply_label_dict(labels, label_dict):
    
    Map raw class labels (e.g. from CM_labels in the CSV) to display labels
    using a dict, e.g. {'Null': 'No Plastic', 'Plastic': 'Plastic Detected'}.

    - Order is preserved (this only relabels, it never reorders the matrix).
    - Any label not present in label_dict is left unchanged.
    - If labels is None or label_dict is falsy, labels is returned as-is.
    
    if not label_dict or labels is None:
        return labels
    return [label_dict.get(lbl, lbl) for lbl in labels]


def plot_confusion_matrix(
    matrix,
    labels=None,
    title=None,
    title_line1=None,
    title_line2=None,
    figsize=(6, 5),
    cmap='Blues',
    normalize=False,
    fmt=None,
    # --- customizable text styling ---
    font_family='sans-serif',
    title_fontsize=16,
    title_fontweight='bold',
    title1_fontsize=16,
    title1_fontweight='bold',
    title2_fontsize=12,
    title2_fontweight='normal',
    title_gap=0.06,          # vertical gap between the two title lines (axes fraction)
    title_pad=0.10,          # gap between line2 and the top of the heatmap (axes fraction)
    label_fontsize=12,       # axis titles ("Predicted", "Actual")
    tick_fontsize=11,        # class name labels on the ticks
    annot_fontsize=13,       # numbers inside the cells
    cbar=False,
    # --- output ---
    save_path=None,
    dpi=300,
):
    
    Plot (and optionally save) a single pretty confusion matrix with seaborn.

    Two title modes:
    - Pass `title` for a single-line title (original behavior).
    - Pass `title_line1` (e.g. the question, bold/larger) and/or `title_line2`
      (e.g. "LR, n = 192, Accuracy = 91.0%", smaller) for a two-line title.
      If both title_line1/2 and title are given, the two-line version wins.

    Parameters
    ----------
    matrix : np.ndarray
        The confusion matrix (rows = actual, cols = predicted).
    labels : list, optional
        Class labels, used for both axes. To rename these for display without
        reordering the matrix, use `apply_label_dict()` before calling this,
        or use the `label_dict` support in `save_all_confusion_matrices`.
    normalize : bool
        If True, color by row-normalized percentages (annotations still show raw counts + %).
    save_path : str, optional
        If given, saves the figure to this path at `dpi` resolution.
    dpi : int
        Resolution for saved PNG (300 = print quality, 600 = very high res).
    
    matrix = np.asarray(matrix)

    if normalize:
        with np.errstate(all='ignore'):
            norm_data = matrix.astype(float) / matrix.sum(axis=1, keepdims=True)
        # annotate with "count\n(pct%)"
        annot_labels = np.array([
            f"{count}\n({pct:.1%})" for count, pct in zip(matrix.flatten(), norm_data.flatten())
        ]).reshape(matrix.shape)
        plot_data = norm_data
        fmt = ''
    else:
        plot_data = matrix
        annot_labels = matrix
        fmt = fmt or 'd'

    fig, ax = plt.subplots(figsize=figsize)

    with plt.rc_context({'font.family': font_family}):
        sns.heatmap(
            plot_data,
            annot=annot_labels,
            fmt=fmt,
            cmap=cmap,
            cbar=cbar,
            square=True,
            linewidths=0.5,
            linecolor='white',
            xticklabels=labels if labels is not None else 'auto',
            yticklabels=labels if labels is not None else 'auto',
            annot_kws={'fontsize': annot_fontsize},
            ax=ax,
        )

        if title_line1 or title_line2:
            # Two-line title, stacked above the axes, each with its own styling.
            y_line1 = 1.0 + title_pad + title_gap
            y_line2 = 1.0 + title_pad
            if title_line1:
                ax.text(
                    0.5, y_line1, str(title_line1),
                    transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=title1_fontsize, fontweight=title1_fontweight,
                    fontfamily=font_family,
                )
            if title_line2:
                ax.text(
                    0.5, y_line2, str(title_line2),
                    transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=title2_fontsize, fontweight=title2_fontweight,
                    fontfamily=font_family,
                )
        elif title:
            ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight, pad=14)

        ax.set_xlabel('Predicted', fontsize=label_fontsize)
        ax.set_ylabel('Actual', fontsize=label_fontsize)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    if save_path:
        # bbox_inches='tight' ensures the text placed above the axes (for the
        # two-line title) isn't clipped out of the saved PNG.
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def save_all_confusion_matrices(
    results,
    output_dir='confusion_matrices',
    dpi=300,
    file_prefix='cm_',
    file_suffix='.png',
    metric_info=None,
    **plot_kwargs,
):

    Save one high-res PNG per Question's combined confusion matrix.

    Parameters
    ----------
    results : dict
        Output of combine_confusion_matrices().
    output_dir : str
        Folder to save PNGs into (created if it doesn't exist).
    dpi : int
        Resolution for saved PNGs.
    file_prefix / file_suffix : str
        Used to build filenames: f"{prefix}{question}{suffix}"
    metric_info : dict, optional
        Maps question -> {'algorithm': ..., 'test_metric_text': ..., 'label_dict': ...}.
        Overrides any 'algorithm' / 'test_metric_text' already present in
        `results` (e.g. pulled from CSV columns via combine_confusion_matrices).

        'label_dict' is optional per-question and lets you rename the raw
        class labels for display (e.g. CM_labels of ['Null', 'Plastic'] ->
        ['No Plastic', 'Plastic Detected']) WITHOUT changing the underlying
        matrix order/values. Any label not in the dict is left unchanged.

        Example:
            {
                'Q1': {
                    'algorithm': 'LR',
                    'test_metric_text': 'Accuracy = 91.0%',
                    'label_dict': {'Null': 'No Plastic', 'Plastic': 'Plastic Detected'},
                },
                'Q2': {
                    'algorithm': 'RF',
                    'test_metric_text': 'F1 = 0.87',
                    'label_dict': {'0': 'Negative', '1': 'Positive'},
                },
            }
    plot_kwargs : dict
        Passed through to plot_confusion_matrix (fonts, sizes, cmap, normalize, etc).

    Returns
    -------
    list of saved file paths

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    metric_info = metric_info or {}

    for question, info in results.items():
        safe_question = re.sub(r'[^\w\-]', '_', str(question))  # sanitize for filenames
        filename = f"{file_prefix}{safe_question}{file_suffix}"
        save_path = os.path.join(output_dir, filename)

        n = int(info['matrix'].sum())
        override = metric_info.get(question, {})
        algorithm = override.get('algorithm', info.get('algorithm'))
        test_metric_text = override.get('test_metric_text', info.get('test_metric_text'))
        title_str = override.get('bolded_title', info.get('bolded_title'))

        # Rename display labels if a label_dict was supplied for this question,
        # without touching matrix order (which is fixed by CM_labels order).
        label_dict = override.get('label_dict')
        display_labels = apply_label_dict(info['labels'], label_dict)

        line2_parts = []
        if algorithm:
            line2_parts.append(str(algorithm))
        line2_parts.append(f"n = {n}")
        if test_metric_text:
            line2_parts.append(str(test_metric_text))
        title_line2 = ", ".join(line2_parts)

        plot_confusion_matrix(
            info['matrix'],
            labels=display_labels,
            title_line1=title_str,
            title_line2=title_line2,
            save_path=save_path,
            dpi=dpi,
            **plot_kwargs,
        )
        saved_paths.append(save_path)
        print(f"Saved: {save_path}")

    return saved_paths


def CM_Master(csv_filename, metric_info=None, algorithm_col=None, test_metric_col=None):

    Parameters
    ----------
    metric_info : dict, optional
        See save_all_confusion_matrices. Use this when algorithm/metric
        aren't columns in your CSV, or to override CSV values, and/or to
        supply a per-question 'label_dict' to rename display labels.
    algorithm_col, test_metric_col : str, optional
        Column names in the CSV to pull algorithm / metric text from
        automatically (see combine_confusion_matrices).

    results = combine_confusion_matrices(
        csv_filename,
        algorithm_col=algorithm_col,
        test_metric_col=test_metric_col,
    )

    saved_files = save_all_confusion_matrices(
        results,
        output_dir='confusion_matrices',
        dpi=300,              # bump to 600 for very high-res
        figsize=(6, 5),
        font_family='sans-serif',
        title1_fontsize=16,
        title1_fontweight='bold',
        title2_fontsize=12,
        title2_fontweight='normal',
        label_fontsize=13,
        tick_fontsize=12,
        annot_fontsize=14,
        cmap='Blues',
        normalize=False,      # set True to also show row percentages
        metric_info=metric_info,
    )

    return saved_files"""
import ast
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_cm_string(cm_str):
    """Parse a confusion matrix string like '[[ 0  8]\n [ 1 11]]' into a numpy array."""
    cleaned = cm_str.strip()
    cleaned = re.sub(r'(?<=\d)\s+(?=-?\d)', ', ', cleaned)
    cleaned = re.sub(r'\]\s*\[', '], [', cleaned)
    return np.array(ast.literal_eval(cleaned))


def parse_labels(raw_labels):
    """Parse a labels string like "['Null' 'Plastic']" into a list."""
    if not isinstance(raw_labels, str):
        return raw_labels
    cleaned = re.sub(r"'\s+'", "', '", raw_labels.strip())
    return ast.literal_eval(cleaned)


def combine_confusion_matrices(csv_path, question_col='Question', cm_col='CM',
                                labels_col='CM_labels',
                                algorithm_col='Algorithm', test_metric_col=None):
    """
    Combine confusion matrices, grouped by question AND (if available) algorithm.

    Parameters
    ----------
    algorithm_col : str, optional
        Name of a CSV column holding the algorithm name (e.g. 'LR', 'RF').
        If given and present in the CSV, matrices are grouped and combined
        separately per (question, algorithm) pair -- i.e. you get one
        combined CM per algorithm per question. If the column is missing
        (or algorithm_col=None), falls back to grouping by question only
        (original behavior).
    test_metric_col : str, optional
        Name of a CSV column holding a pre-formatted metric string
        (e.g. 'Accuracy = 91.0%'). If given and present, the first value
        per group is stored in results[key]['test_metric_text']. Used only
        by the 'metric_info' title_mode in save_all_confusion_matrices --
        for 'auto' title_mode, pass an external `metric_results_dict`
        instead (see save_all_confusion_matrices).

    Returns
    -------
    dict
        If algorithm grouping is active, keys are (question, algorithm)
        tuples. Otherwise keys are just the question. Each value has:
        'question', 'algorithm', 'matrix', 'labels', 'n_rows',
        'test_metric_text'.
    """
    df = pd.read_csv(csv_path)
    df['_cm_parsed'] = df[cm_col].apply(parse_cm_string)

    has_algorithm = bool(algorithm_col) and algorithm_col in df.columns
    group_cols = [question_col, algorithm_col] if has_algorithm else [question_col]

    results = {}
    for group_key, group in df.groupby(group_cols):
        if has_algorithm:
            question, algorithm = group_key
        else:
            question, algorithm = group_key, None

        matrices = list(group['_cm_parsed'])
        shapes = {m.shape for m in matrices}
        if len(shapes) > 1:
            label = f"'{question}'" + (f" / algorithm '{algorithm}'" if has_algorithm else "")
            raise ValueError(f"Question {label} has mismatched CM shapes: {shapes}")

        combined = np.sum(matrices, axis=0)

        labels = None
        if labels_col in group.columns:
            labels = parse_labels(group[labels_col].iloc[0])

        test_metric_text = None
        if test_metric_col and test_metric_col in group.columns:
            test_metric_text = group[test_metric_col].iloc[0]

        key = (question, algorithm) if has_algorithm else question
        results[key] = {
            'question': question,
            'algorithm': algorithm,
            'matrix': combined,
            'labels': labels,
            'n_rows': len(group),
            'test_metric_text': test_metric_text,
        }

    return results


def apply_label_dict(labels, label_dict):
    """
    Map raw class labels (e.g. from CM_labels in the CSV) to display labels
    using a dict, e.g. {'Null': 'No Plastic', 'Plastic': 'Plastic Detected'}.

    - Order is preserved (this only relabels, it never reorders the matrix).
    - Any label not present in label_dict is left unchanged.
    - If labels is None or label_dict is falsy, labels is returned as-is.
    """
    if not label_dict or labels is None:
        return labels
    return [label_dict.get(lbl, lbl) for lbl in labels]


def plot_confusion_matrix(
    matrix,
    labels=None,
    title=None,
    title_line1=None,
    title_line2=None,
    figsize=(6, 5),
    cmap='Blues',
    normalize=False,
    fmt=None,
    # --- customizable text styling ---
    font_family='sans-serif',
    title_fontsize=16,
    title_fontweight='bold',
    title1_fontsize=16,
    title1_fontweight='bold',
    title2_fontsize=12,
    title2_fontweight='normal',
    title_gap=0.06,          # vertical gap between the two title lines (axes fraction)
    title_pad=0.10,          # gap between line2 and the top of the heatmap (axes fraction)
    label_fontsize=12,       # axis titles ("Predicted", "Actual")
    tick_fontsize=11,        # class name labels on the ticks
    annot_fontsize=13,       # numbers inside the cells
    cbar=False,
    # --- output ---
    save_path=None,
    dpi=300,
):
    """
    Plot (and optionally save) a single pretty confusion matrix with seaborn.

    Two title modes:
    - Pass `title` for a single-line title (original behavior).
    - Pass `title_line1` (e.g. the question, bold/larger) and/or `title_line2`
      (e.g. "LR, n = 192, Accuracy = 91.0%", smaller) for a two-line title.
      If both title_line1/2 and title are given, the two-line version wins.

    Parameters
    ----------
    matrix : np.ndarray
        The confusion matrix (rows = actual, cols = predicted).
    labels : list, optional
        Class labels, used for both axes. To rename these for display without
        reordering the matrix, use `apply_label_dict()` before calling this,
        or use the `label_dict` support in `save_all_confusion_matrices`.
    normalize : bool
        If True, color by row-normalized percentages (annotations still show raw counts + %).
    save_path : str, optional
        If given, saves the figure to this path at `dpi` resolution.
    dpi : int
        Resolution for saved PNG (300 = print quality, 600 = very high res).
    """
    matrix = np.asarray(matrix)

    if normalize:
        with np.errstate(all='ignore'):
            norm_data = matrix.astype(float) / matrix.sum(axis=1, keepdims=True)
        # annotate with "count\n(pct%)"
        annot_labels = np.array([
            f"{count}\n({pct:.1%})" for count, pct in zip(matrix.flatten(), norm_data.flatten())
        ]).reshape(matrix.shape)
        plot_data = norm_data
        fmt = ''
    else:
        plot_data = matrix
        annot_labels = matrix
        fmt = fmt or 'd'

    fig, ax = plt.subplots(figsize=figsize)

    with plt.rc_context({'font.family': font_family}):
        sns.heatmap(
            plot_data,
            annot=annot_labels,
            fmt=fmt,
            cmap=cmap,
            cbar=cbar,
            square=True,
            linewidths=0.5,
            linecolor='white',
            xticklabels=labels if labels is not None else 'auto',
            yticklabels=labels if labels is not None else 'auto',
            annot_kws={'fontsize': annot_fontsize},
            ax=ax,
        )

        if title_line1 or title_line2:
            # Two-line title, stacked above the axes, each with its own styling.
            y_line1 = 1.0 + title_pad + title_gap
            y_line2 = 1.0 + title_pad
            if title_line1:
                ax.text(
                    0.5, y_line1, str(title_line1),
                    transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=title1_fontsize, fontweight=title1_fontweight,
                    fontfamily=font_family,
                )
            if title_line2:
                ax.text(
                    0.5, y_line2, str(title_line2),
                    transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=title2_fontsize, fontweight=title2_fontweight,
                    fontfamily=font_family,
                )
        elif title:
            ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight, pad=14)

        ax.set_xlabel('Predicted', fontsize=label_fontsize)
        ax.set_ylabel('Actual', fontsize=label_fontsize)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    if save_path:
        # bbox_inches='tight' ensures the text placed above the axes (for the
        # two-line title) isn't clipped out of the saved PNG.
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def save_all_confusion_matrices(
    results,
    output_dir='confusion_matrices',
    dpi=300,
    file_prefix='cm_',
    file_suffix='.png',
    metric_info=None,
    title_mode='metric_info',
    metric_results_dict=None,
    metric_value_fmt='{:.1%}',
    **plot_kwargs,
):
    """
    Save one high-res PNG per Question (or per Question+Algorithm, if the
    results came from combine_confusion_matrices with algorithm grouping).

    Parameters
    ----------
    results : dict
        Output of combine_confusion_matrices(). Keys are either a question
        string, or (question, algorithm) tuples when algorithm grouping was
        used.
    output_dir : str
        Folder to save PNGs into (created if it doesn't exist).
    dpi : int
        Resolution for saved PNGs.
    file_prefix / file_suffix : str
        Used to build filenames:
            f"{prefix}{question}{suffix}"                 (no algorithm)
            f"{prefix}{question}_{algorithm}{suffix}"      (with algorithm)
    title_mode : {'metric_info', 'auto'}
        Controls how the chart title (title_line1 / title_line2) is built.

        - 'metric_info' (default): use the manual overrides in `metric_info`
          for 'bolded_title', 'algorithm', and 'test_metric_text', falling
          back to whatever combine_confusion_matrices() found in the CSV
          (original behavior).
        - 'auto': skip the manual title overrides entirely and set
          title_line1 = question, title_line2 = "{algorithm}, {mean} ± {sem}",
          where mean/sem are pulled from `metric_results_dict` for this
          (question, algorithm) pair and formatted with `metric_value_fmt`.

        Either way, class-label renaming (label_dict) is always resolved by
        question only (see below) -- it never varies by algorithm.
    metric_results_dict : dict, optional
        Required for 'auto' title_mode. Externally computed stats keyed by
        question then algorithm, e.g.:
            {
                'AS': {
                    'LDA': {'mean': [0.821], 'std': [0.129], 'sem': [0.037]},
                    'RF':  {'mean': [0.87],  'std': [0.09],  'sem': [0.021]},
                },
                ...
            }
        'mean' and 'sem' can be plain numbers or single-item lists/arrays
        (either is handled) -- the first element is used. Ignored entirely
        in 'metric_info' title_mode.
    metric_value_fmt : str, optional
        Format string applied to both mean and sem in 'auto' title_mode,
        e.g. '{:.1%}' (default) -> '82.1%', '{:.3f}' -> '0.821'. Ignored in
        'metric_info' mode.
    metric_info : dict, optional
        Overrides for 'bolded_title' / 'label_dict' (used in either title
        mode), and 'algorithm' / 'test_metric_text' (used only in
        'metric_info' title mode). Keys can be the same key used in
        `results` (question, or (question, algorithm) tuple) or just the
        plain question string.

        Note: 'label_dict' and 'bolded_title' are always looked up by
        question only (any algorithm-specific tuple key you provide for
        these is ignored), since labeling/titling-by-question should stay
        consistent across algorithms for the same question.

        'label_dict' lets you rename the raw class labels for display
        (e.g. CM_labels of ['Null', 'Plastic'] -> ['No Plastic',
        'Plastic Detected']) WITHOUT changing the underlying matrix
        order/values. Any label not in the dict is left unchanged.

        Example (metric_info title mode):
            {
                'Q1': {'label_dict': {'Null': 'No Plastic', 'Plastic': 'Plastic Detected'}},
                ('Q1', 'LR'): {'test_metric_text': 'Accuracy = 91.0%'},
                ('Q1', 'RF'): {'test_metric_text': 'Accuracy = 88.4%'},
            }
    plot_kwargs : dict
        Passed through to plot_confusion_matrix (fonts, sizes, cmap, normalize, etc).

    Returns
    -------
    list of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    metric_info = metric_info or {}
    metric_results_dict = metric_results_dict or {}

    def _first(val):
        """Unwrap a single-item list/tuple/array (or np scalar) to a plain float."""
        if val is None:
            return None
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) == 0:
                return None
            val = val[0]
        return float(val)

    for key, info in results.items():
        question = info['question']
        algorithm = info.get('algorithm')

        safe_question = re.sub(r'[^\w\-]', '_', str(question))  # sanitize for filenames
        if algorithm:
            safe_algorithm = re.sub(r'[^\w\-]', '_', str(algorithm))
            filename = f"{file_prefix}{safe_question}_{safe_algorithm}{file_suffix}"
        else:
            filename = f"{file_prefix}{safe_question}{file_suffix}"
        save_path = os.path.join(output_dir, filename)

        n = int(info['matrix'].sum())

        # Labeling (and the bolded question title) is always resolved by
        # question only, regardless of title_mode or algorithm, so the same
        # question always gets the same class-label renaming.
        question_override = metric_info.get(question, {})
        label_dict = question_override.get('label_dict')
        display_labels = apply_label_dict(info['labels'], label_dict)
        title_str = question_override.get('bolded_title', info.get('bolded_title', question))

        if title_mode == 'auto':
            stats = metric_results_dict.get(question, {}).get(algorithm, {})
            mean_val = _first(stats.get('mean'))
            sem_val = _first(stats.get('sem'))

            metric_str = None
            if mean_val is not None:
                mean_str = metric_value_fmt.format(mean_val)
                if sem_val is not None:
                    sem_str = metric_value_fmt.format(sem_val)
                    metric_str = f"{mean_str} \u00b1 {sem_str}"
                else:
                    metric_str = mean_str

            line2_parts = []
            if algorithm:
                line2_parts.append(str(algorithm))
            line2_parts.append(f"n = {n}")
            if metric_str:
                line2_parts.append(metric_str)
            title_line2 = ", ".join(line2_parts)
        elif title_mode == 'metric_info':
            # Algorithm/metric-text overrides CAN be algorithm-specific here,
            # so look up the exact (question, algorithm) key first.
            override = metric_info.get(key) or question_override
            algorithm_display = override.get('algorithm', algorithm)
            test_metric_text = override.get('test_metric_text', info.get('test_metric_text'))

            line2_parts = []
            if algorithm_display:
                line2_parts.append(str(algorithm_display))
            line2_parts.append(f"n = {n}")
            if test_metric_text:
                line2_parts.append(str(test_metric_text))
            title_line2 = ", ".join(line2_parts)
        else:
            raise ValueError(f"Unknown title_mode: {title_mode!r} (expected 'metric_info' or 'auto')")

        plot_confusion_matrix(
            info['matrix'],
            labels=display_labels,
            title_line1=title_str,
            title_line2=title_line2,
            save_path=save_path,
            dpi=dpi,
            **plot_kwargs,
        )
        saved_paths.append(save_path)
        print(f"Saved: {save_path}")

    return saved_paths


def CM_Master(csv_filename, metric_info=None, algorithm_col='Algorithm', test_metric_col=None,
              title_mode='metric_info', metric_results_dict=None, metric_value_fmt='{:.1%}'):
    """
    Parameters
    ----------
    metric_info : dict, optional
        See save_all_confusion_matrices. Use this to override the bolded
        question title / label_dict (always by question), and -- in
        'metric_info' title_mode -- algorithm / metric text per
        (question, algorithm).
    algorithm_col : str, optional
        Column name in the CSV holding the algorithm per row (e.g. 'LR',
        'RF'). When present, one combined CM is produced per algorithm per
        question. Pass None (or a column that isn't present) to fall back
        to one CM per question, combining across all algorithms.
    test_metric_col : str, optional
        Column name in the CSV to pull a pre-formatted metric string from
        automatically (used in 'metric_info' title_mode).
    title_mode : {'metric_info', 'auto'}
        'metric_info' (default): title built from metric_info overrides /
        CSV columns, as before -- "{algorithm}, n = {n}, {test_metric_text}".
        'auto': skip metric_info-based titling and use
        "{algorithm}, {mean} ± {sem}" instead, pulled from
        `metric_results_dict`. Class-label renaming still follows
        `metric_info` by question either way.
    metric_results_dict : dict, optional
        Required for 'auto' title_mode. Externally computed stats keyed by
        question then algorithm, e.g.
        {'AS': {'LDA': {'mean': [0.821], 'sem': [0.037]}}}. See
        save_all_confusion_matrices for the full shape.
    metric_value_fmt : str, optional
        Format string for mean/sem in 'auto' title_mode, e.g. '{:.1%}'
        (default) -> '82.1%', '{:.3f}' -> '0.821'.
    """
    results = combine_confusion_matrices(
        csv_filename,
        algorithm_col=algorithm_col,
        test_metric_col=test_metric_col,
    )

    saved_files = save_all_confusion_matrices(
        results,
        output_dir='confusion_matrices',
        dpi=300,              # bump to 600 for very high-res
        figsize=(6, 5),
        font_family='sans-serif',
        title1_fontsize=16,
        title1_fontweight='bold',
        title2_fontsize=12,
        title2_fontweight='normal',
        label_fontsize=13,
        tick_fontsize=12,
        annot_fontsize=14,
        cmap='Blues',
        normalize=False,      # set True to also show row percentages
        metric_info=metric_info,
        title_mode=title_mode,
        metric_results_dict=metric_results_dict,
        metric_value_fmt=metric_value_fmt,
    )

    return saved_files

### Main Code ###

PreprocessMetricValues("HPC_results_cm_all.csv")
result_dict = ProcessMetricValuesRedo("HPC_results_cm_all.csv")
metric_info = {
    'Detection': {
        'algorithm': 'SVM-R',
        'test_metric_text': 'F1 Score = 90.9%',
        'label_dict': {'Null': "Unspiked", 'Plastic': 'Spiked'},
        "bolded_title": "A. Plastic Presence",
    },
    'Type_NoDI': {
        'algorithm': 'SVM-R',
        'test_metric_text': 'Accuracy = 82.1%',
        'label_dict': {'VC': "PVC", 'MC': 'PMMA', 'SC': "PS", "LC": "LDPE", "TC": "PET"},
        "bolded_title": "B. Plastic Type",
    },
    'AS': {
        'algorithm': 'LR',
        'test_metric_text': 'F1 Score = 88.0%',
        'label_dict': {'Null': "Unspiked", 'Plastic': 'Spiked'},
        'bolded_title': "B. In ammonium sulfate",
    },
    'CB': {
        'algorithm': 'SVM-R',
        'test_metric_text': 'F1 Score = 80.8%',
        'label_dict': {'Null': "Unspiked", 'Plastic': 'Spiked'},
        'bolded_title': "C. In black carbon",
    },
    'BW': {
        'algorithm': 'LDA',
        'test_metric_text': 'Accuracy = 84.4%',
        'label_dict': {'BC': "Spiked\n1", 'CC': 'Spiked\n2', "BW1": "Unspiked\n1", "BW2": "Unspiked\n2"},
        'bolded_title': "A. In bottled water",
    },
    'LC': {
        'algorithm': 'SVM-R',
        'test_metric_text': 'F1 Score = 93.4%',
        'label_dict': {'Null': "Unspiked", 'Plastic': 'Spiked'},
        'bolded_title': "D. With lower concentrations",
    },
    'NS': {
        'algorithm': 'LR',
        'test_metric_text': 'F1 Score = 73.5%',
        'label_dict': {'Null': "Unspiked", 'Plastic': 'Spiked'},
        'bolded_title': "Nanospheres Presence",
    },
    'BW_2': {
        'algorithm': 'SVM-R',
        'test_metric_text': 'F1 Score = 89.2%',
        'label_dict': {'Null': "Unspiked", 'Plastic': 'Spiked'},
        'bolded_title': "A. In bottled water",
    },
}

CM_Master('HPC_results_cm_all.csv', title_mode='auto', metric_info=metric_info, metric_results_dict=result_dict, metric_value_fmt='{:.1%}', )



