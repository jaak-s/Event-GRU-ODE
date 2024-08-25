import sklearn.metrics
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import torch
import json

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def groupby(df, col, fn):
    df.sort_values(by=col, inplace=True)
    tasks, task_loc = np.unique(df[col].values, return_index=True)
    task_loc = task_loc[1:]
    starts   = np.concatenate([[0], task_loc])
    ends     = np.concatenate([task_loc, [df.shape[0]]])
    result   = {}
    for task, start, end in zip(tasks, starts, ends):
        result[task] = fn(df.iloc[start:end])
    result = pd.concat(result, axis=0)
    result.index = tasks
    result.index.name = col
    return result

def groupby_fast(df, col, fn):
    col_min = df[col].values.min()
    col_max = df[col].values.max()
    result   = {}
    values   = list(range(col_min, col_max+1))
    if col_min == col_max:
        result[col_min] = fn(df)
    else:
        for val in range(col_min, col_max+1):
            g = df[df[col].values == val]
            result[val] = fn(g)
    result_df = pd.concat(result, axis=0)
    result_df.index = values
    result_df.index.name = col
    return result_df


def all_metrics(y_true, y_score, tpr_levels):
    if len(y_true) <= 1 or (y_true[0] == y_true).all():
        no_data = True
    else:
        no_data = False
    #    df = pd.DataFrame({"roc_auc": [np.nan], "auc_pr": [np.nan], "avg_prec": [np.nan], "max_f1": [np.nan]})
    #    return df
    n_total = len(y_true)
    n_pos   = (y_true == 1).sum()
    n_neg   = n_total - n_pos

    metrics = {}

    fpr, tpr, auc_thresholds = sklearn.metrics.roc_curve(y_true, y_score)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true = y_true, probas_pred = y_score)

    fpr_levels  = np.array([0.01, 0.05, 0.1, 0.2, 0.7])
    fpr_idx     = np.searchsorted(fpr, fpr_levels)
    thr         = auc_thresholds[ fpr_idx ]
    precs_fpr   = precision[np.searchsorted(thresholds, thr)]
    recalls_fpr = tpr[ fpr_idx ]
    prevalence  = n_pos / n_total

    for fpr_level, prec, rec in zip(fpr_levels, precs_fpr, recalls_fpr):
        metrics[f"prec@fpr{fpr_level}"]   = prec
        metrics[f"recall@fpr{fpr_level}"] = rec
        metrics[f"f1@fpr{fpr_level}"]  = 2 * (prec * rec) / (prec + rec)
        ## using:
        ## prec = (rec x prevalence) / [ (rec x prevalence) + (fpr x (1 – prevalence)) ]
        fp_rate = rec * (1 / prec - 1) / (1 / prevalence - 1)
        metrics[f"acc@fpr{fpr_level}"] = (rec * n_pos + (n_total - fp_rate * n_neg)) / n_total

    metrics["roc_auc"] = sklearn.metrics.auc(fpr, tpr)
    metrics["auc_pr"]  = sklearn.metrics.auc(x = recall, y = precision)
    metrics["avg_prec"] = sklearn.metrics.average_precision_score(
          y_true  = y_true,
          y_score = y_score)

    higher        = recall > tpr_levels[:, None]
    recall_idx    = [np.where(h)[0][-1] for h in higher]

    for i, rec in enumerate(tpr_levels):
        prec = precision[recall_idx[i]]
        metrics[f"prec@recall{rec}"] = prec

    for i, rec in enumerate(tpr_levels):
        prec = precision[recall_idx[i]]
        metrics[f"f1@recall{rec}"]   = 2 * (prec * rec) / (prec + rec)

    for i, rec in enumerate(tpr_levels):
        threshold = thresholds[recall_idx[i]]
        metrics[f"acc@recall{rec}"]  = ((y_score >= threshold) == y_true).mean()

    metrics = pd.DataFrame({k: [v] for k, v in metrics.items()})
    return metrics

def compute_metrics(cols, y_true, y_score, num_tasks, tpr_levels=None):
    if tpr_levels is None:
        tpr_levels = np.array([0.99, 0.95, 0.9, 0.8, 0.3])
        
    if len(cols) < 1:
        return pd.DataFrame({
            "roc_auc":  np.nan,
            "auc_pr":   np.nan,
            "avg_prec": np.nan,
            "max_f1":   np.nan}, index=np.arange(num_tasks))

    df = pd.DataFrame({"task": cols, "y_true": y_true, "y_score": y_score})

    metrics = groupby_fast(df, "task", lambda g:
        all_metrics(
            y_true     = g.y_true.values,
            y_score    = g.y_score.values,
            tpr_levels = tpr_levels,
        )
    )

    return metrics

def vec_to_str(aucs, num_digits=4):
    """Returns string of aucs"""
    strs = []
    for v in aucs:
        strs.append(f"{v:.4f}")
    return ", ".join(strs)

def compute_batch_metrics(output, target, mask):
    tx, px, cols = torch.where(mask)
    res = eg.compute_metrics(cols.cpu().numpy(), target.cpu().numpy(), output.detach().cpu().numpy(), num_tasks=mask.shape[-1])
    return res

def load_results(filename):
    with open(filename, "r") as f:
        data = json.load(f)
        data["results_va"]["metrics"] = pd.read_json(data["results_va"]["metrics"])
        return data
