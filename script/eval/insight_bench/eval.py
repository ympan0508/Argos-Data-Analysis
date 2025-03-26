# Modified from:
# https://github.com/ServiceNow/insight-bench/blob/main/insightbench/benchmarks.py

import argparse
import json
import os

import numpy as np
from metrics import (
    compute_g_eval,
    compute_g_eval_with_cot,
    compute_rouge_score,
)


def eval_summary(pred_summary, dataset_dict, metric="rouge1"):
    scores = []

    goal = dataset_dict["metadata"]["goal"]
    gt_summary = dataset_dict["summary"]

    if metric == "rouge1":
        score_summary = compute_rouge_score(pred_summary, gt_summary)
        scores.append(score_summary)
    elif metric == "g_eval":
        score_summary = compute_g_eval(pred_summary, gt_summary)
        scores.append(score_summary)
    elif metric == "g_eval_cot":
        score_summary = compute_g_eval_with_cot(pred_summary, gt_summary, goal)
        scores.append(score_summary)
    else:
        raise ValueError(
            f"Unsupported metric: {metric}. "
            "Supported metrics are 'rouge1', 'g_eval', 'g_eval_cot'.")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path', default=os.path.join(
            os.path.abspath(os.path.dirname(__file__)), 'notebooks'))
    parser.add_argument('--pred', default='example.json')
    parser.add_argument('--metric', default='rouge1')
    args = parser.parse_args()

    with open(args.pred, "r") as f:
        answer = json.load(f)

    result = []

    for i in range(len(answer)):
        with open(os.path.join(args.dataset_path, f"flag-{i+1}.json")) as f:
            dataset_dict = json.load(f)
        pred_summary = answer[i]["report"]

        if pred_summary:
            result.append(eval_summary(
                pred_summary, dataset_dict, metric=args.metric))
        else:
            result.append(None)
    print(np.mean([x for x in result if x is not None], axis=0))
