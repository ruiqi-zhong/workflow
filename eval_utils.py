import json
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import spearmanr
import numpy as np


def rank_of_each_element(ls):
    sorted_ls = sorted(ls)
    x2index = {x: i for i, x in enumerate(sorted_ls)}
    return np.array([x2index[x] / len(ls) for x in ls])

def evaluate_pred_path(pred_path):
    if type(pred_path) == str:
        preds = json.load(open(pred_path))
        pred_scores = [p['pred_score'] for p in preds]
        pred_scores = np.array(pred_scores)
        pred_scores = rank_of_each_element(pred_scores)
    else:
        pred_paths = pred_path
        preds = json.load(open(pred_paths[0]))
        pred_scores_l = [[x['pred_score'] for x in json.load(open(p))] for p in pred_paths]
        pred_scores_l = [rank_of_each_element(x) for x in pred_scores_l]
        pred_scores = np.mean(pred_scores_l, axis=0)

    orig_scores = [p['orig_d']['target'] for p in preds]
    discrete_gold_label = [1 if s > 0.5 else 0 for s in orig_scores]
    discrete_pred_label = [1 if s > 0.5 else 0 for s in pred_scores]
    optimal_acc, optimal_threshold = 0, 0
    for threshold in np.arange(0.0, 1.0, 0.01):
        acc = accuracy_score(discrete_gold_label, [1 if s > threshold else 0 for s in pred_scores])
        if acc > optimal_acc:
            optimal_acc = acc
            optimal_threshold = threshold

    result_dict = {
        'accuracy': accuracy_score(discrete_gold_label, discrete_pred_label),
        'auc': roc_auc_score(discrete_gold_label, pred_scores),
        'spearman': spearmanr(orig_scores, pred_scores),
        'optimal_acc': optimal_acc,
        'optimal_threshold': optimal_threshold
    }
    return result_dict
