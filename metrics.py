"""Original retrieval metric conventions; no training dependencies."""
import time
import numpy as np

def rank_and_evaluate(query_code, database_code, query_label, database_label, args):
    query_code_f = query_code.astype(np.float32, copy=False)
    database_code_f = database_code.astype(np.float32, copy=False)
    query_label = np.asarray(query_label)
    database_label = np.asarray(database_label)
    topk_values = [
        value for value in parse_int_list(args.topk) if value > 0
    ]
    recall_grid = np.linspace(0.0, 1.0, args.pr_points)
    precision_sum = np.zeros(len(topk_values), dtype=np.float64)
    pr_sum = np.zeros(args.pr_points, dtype=np.float64)
    ap_all_sum = 0.0
    ap_50_sum = 0.0
    valid_queries = 0
    distance_time = 0.0
    ranking_time = 0.0
    metric_time = 0.0

    for query_index in range(query_code.shape[0]):
        start = time.perf_counter()
        distance = 0.5 * (
            args.nbit
            - np.dot(query_code_f[query_index], database_code_f.T)
        )
        distance_time += time.perf_counter() - start

        start = time.perf_counter()
        order = np.argsort(distance, kind="stable")
        ranking_time += time.perf_counter() - start

        start = time.perf_counter()
        relevant = (
            np.dot(query_label[query_index], database_label.T) > 0
        ).astype(np.float64)[order]
        total_relevant = int(relevant.sum())
        if total_relevant > 0:
            valid_queries += 1
            cumulative = np.cumsum(relevant)
            rank = np.arange(1, len(relevant) + 1, dtype=np.float64)
            precision = cumulative / rank
            positive = np.flatnonzero(relevant)
            ap_all_sum += float(np.mean(precision[positive]))
            positive_50 = positive[positive < 50]
            if positive_50.size:
                ap_50_sum += float(np.mean(precision[positive_50]))
            for index, topk in enumerate(topk_values):
                cutoff = min(topk, len(relevant))
                precision_sum[index] += relevant[:cutoff].sum() / cutoff
            recall = cumulative / total_relevant
            for index, recall_value in enumerate(recall_grid):
                reached = np.flatnonzero(recall >= recall_value)
                pr_sum[index] += (
                    precision[reached[0]] if reached.size else precision[-1]
                )
        metric_time += time.perf_counter() - start

    denominator = query_code.shape[0]  # Same convention as the uploaded code.
    metrics = {
        "MAP_all": ap_all_sum / denominator,
        "MAP_at_50": ap_50_sum / denominator,
        "valid_queries": valid_queries,
        "total_queries": int(denominator),
    }
    precision_at_k = precision_sum / denominator
    pr_precision = pr_sum / max(valid_queries, 1)
    timings = {
        "hamming_distance_sec": distance_time,
        "ranking_sec": ranking_time,
        "metric_calculation_sec": metric_time,
    }
    return metrics, topk_values, precision_at_k, recall_grid, pr_precision, timings

def parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]
