import json
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import spearmanr
import numpy as np


def seq2seq_eval_path(path):
    data = json.load(open(path))
    old_length = len(data)
    data = [x for x in data if "[Y]" not in x["prompt"]]
    # print("removed", old_length - len(data), "examples")

    labels = [x["orig_d"]["completion"] for x in data]
    preds = [x["generations"][0]["lm_postprocess"] for x in data]
    return np.mean([x == y for x, y in zip(labels, preds)])


def bin_score(scores, num_bins):
    # Calculate bin edges based on percentiles
    bin_edges = np.percentile(scores, np.linspace(0, 100, num_bins))

    # Assign each score to a bin
    binned_scores = np.digitize(scores, bin_edges, right=True)

    # Adjust for scores equal to the maximum bin edge
    binned_scores[binned_scores > num_bins] = num_bins

    return binned_scores


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inverse_sigmoid(x):
    if x > 0.99999:
        x = 0.99999
    return np.log(x / (1 - x))


interval = 0.1
Xs = sigmoid(np.arange(-5, 5, interval))


def round_number(x):
    logit = inverse_sigmoid(x)
    idx = int((logit + 5) / interval)
    if idx < 0:
        idx = 0
    if idx >= len(Xs):
        idx = len(Xs) - 1
    return idx


def rank_of_each_element(ls):
    sorted_ls = sorted(ls)
    x2index = {x: i for i, x in enumerate(sorted_ls)}
    return np.array([x2index[x] / len(ls) for x in ls])


def recalibration_map(orig_score2calibrated_score):
    ys = []
    for x in Xs:
        # find the closest orig score
        closest_orig_score = min(
            orig_score2calibrated_score.keys(),
            key=lambda orig_score: abs(orig_score - x),
        )
        ys.append(orig_score2calibrated_score[closest_orig_score])
    ys[-1] = 1.0
    ys[0] = 0.0
    return ys


def evaluate_preds(preds):
    orig_pred_scores = [p["pred_score"] for p in preds]
    orig_pred_scores = np.array(orig_pred_scores)
    pred_scores = rank_of_each_element(orig_pred_scores)
    orig_scores = [p["orig_d"]["target"] for p in preds]
    discrete_gold_label = [1 if s > 0.5 else 0 for s in orig_scores]
    discrete_pred_label = [1 if s > 0.5 else 0 for s in pred_scores]
    optimal_acc, optimal_threshold, optimal_discrete_pred_label = 0, 0, None
    for threshold in np.arange(0.0, 1.0, 0.01):
        thresholded_discrete_pred_label = [
            1 if s > threshold else 0 for s in pred_scores
        ]
        acc = accuracy_score(discrete_gold_label, thresholded_discrete_pred_label)
        if acc > optimal_acc:
            optimal_acc = acc
            optimal_threshold = threshold
            optimal_discrete_pred_label = thresholded_discrete_pred_label

    result_dict = {
        "accuracy": accuracy_score(discrete_gold_label, discrete_pred_label),
        "auc": roc_auc_score(discrete_gold_label, pred_scores),
        "rmse": np.sqrt(np.mean((np.array(orig_scores) - np.array(pred_scores)) ** 2)),
        "spearman_corr": spearmanr(orig_scores, pred_scores)[0],
        "spearman_p-value": spearmanr(orig_scores, pred_scores)[1],
        "optimal_acc": optimal_acc,
        "optimal_threshold": optimal_threshold,
        "discrete_pred_label": discrete_pred_label,
        "threshold_tuned_discrete_pred_label": optimal_discrete_pred_label,
        "pred_scores": pred_scores,
        "gold_scores": orig_scores,
    }
    return result_dict


def evaluate_pred_path(pred_path):
    if type(pred_path) == str:
        preds = json.load(open(pred_path))
        orig_pred_scores = [p["pred_score"] for p in preds]
        orig_pred_scores = np.array(orig_pred_scores)
        pred_scores = rank_of_each_element(orig_pred_scores)
        orig_score2calibrated_score = {
            orig_score: calibrated_score
            for orig_score, calibrated_score in zip(orig_pred_scores, pred_scores)
        }
        calibration_map = recalibration_map(orig_score2calibrated_score)
    else:
        pred_paths = pred_path
        preds = json.load(open(pred_paths[0]))
        pred_scores_l = [
            [x["pred_score"] for x in json.load(open(p))] for p in pred_paths
        ]
        pred_scores_l = [rank_of_each_element(x) for x in pred_scores_l]
        orig_score2calibrated_score_l = [
            {
                orig_score: calibrated_score
                for orig_score, calibrated_score in zip(orig_pred_scores, pred_scores)
            }
            for orig_pred_scores, pred_scores in zip(pred_scores_l, pred_scores_l)
        ]
        calibration_map = [
            recalibration_map(orig_score2calibrated_score)
            for orig_score2calibrated_score in orig_score2calibrated_score_l
        ]
        pred_scores = np.mean(pred_scores_l, axis=0)

    orig_scores = [p["orig_d"].get("target") for p in preds]
    selected_idxes = [i for i, x in enumerate(orig_scores) if x is not None]
    if len(selected_idxes) < 2:
        result_dict = {
            "accuracy": 0,
            "auc": 0,
            "rmse": 0,
            "spearman_corr": 0,
            "spearman_p-value": 0,
            "optimal_acc": 0,
            "optimal_threshold": 0,
            "discrete_pred_label": [],
            "threshold_tuned_discrete_pred_label": [],
            "pred_scores": [],
            "gold_scores": [],
        }
        return result_dict

    orig_scores = [orig_scores[i] for i in selected_idxes]
    pred_scores = [pred_scores[i] for i in selected_idxes]

    discrete_gold_label = [1 if s > 0.5 else 0 for s in orig_scores]
    discrete_pred_label = [1 if s > 0.5 else 0 for s in pred_scores]
    optimal_acc, optimal_threshold, optimal_discrete_pred_label = 0, 0, None
    for threshold in np.arange(0.0, 1.0, 0.01):
        thresholded_discrete_pred_label = [
            1 if s > threshold else 0 for s in pred_scores
        ]
        acc = accuracy_score(discrete_gold_label, thresholded_discrete_pred_label)
        if acc > optimal_acc:
            optimal_acc = acc
            optimal_threshold = threshold
            optimal_discrete_pred_label = thresholded_discrete_pred_label

    result_dict = {
        "accuracy": accuracy_score(discrete_gold_label, discrete_pred_label),
        "auc": roc_auc_score(discrete_gold_label, pred_scores),
        "rmse": np.sqrt(np.mean((np.array(orig_scores) - np.array(pred_scores)) ** 2)),
        "spearman_corr": spearmanr(orig_scores, pred_scores)[0],
        "spearman_p-value": spearmanr(orig_scores, pred_scores)[1],
        "optimal_acc": optimal_acc,
        "optimal_threshold": optimal_threshold,
        "calibration_map": calibration_map,
        "discrete_pred_label": discrete_pred_label,
        "threshold_tuned_discrete_pred_label": optimal_discrete_pred_label,
        "pred_scores": pred_scores,
        "gold_scores": orig_scores,
    }
    return result_dict
