from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def mean(values: List[float]) -> float:
    return sum(values) / max(len(values), 1)


def variance(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / (len(values) - 1)


def std(values: List[float]) -> float:
    return math.sqrt(variance(values))


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def welch_t_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    n1 = len(a)
    n2 = len(b)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    m1 = mean(a)
    m2 = mean(b)
    v1 = variance(a)
    v2 = variance(b)
    denom = math.sqrt(v1 / n1 + v2 / n2)
    if denom == 0:
        return 0.0, 1.0
    t_stat = (m1 - m2) / denom
    p_value = 2.0 * (1.0 - normal_cdf(abs(t_stat)))
    return t_stat, clamp(p_value, 0.0, 1.0)


def ks_statistic(a: List[float], b: List[float]) -> Tuple[float, float]:
    if not a or not b:
        return 0.0, 1.0
    a_sorted = sorted(a)
    b_sorted = sorted(b)
    i = 0
    j = 0
    n1 = len(a_sorted)
    n2 = len(b_sorted)
    d = 0.0
    while i < n1 and j < n2:
        if a_sorted[i] <= b_sorted[j]:
            i += 1
        else:
            j += 1
        d = max(d, abs(i / n1 - j / n2))
    d = max(d, abs(1.0 - j / n2), abs(i / n1 - 1.0))
    n_eff = n1 * n2 / (n1 + n2)
    p_value = ks_p_value(d, n_eff)
    return d, p_value


def ks_p_value(d: float, n_eff: float) -> float:
    if n_eff <= 0:
        return 1.0
    x = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * d
    if x <= 0:
        return 1.0
    summation = 0.0
    for k in range(1, 100):
        term = (-1) ** (k - 1) * math.exp(-2.0 * (k * x) ** 2)
        summation += term
        if abs(term) < 1e-8:
            break
    return clamp(2.0 * summation, 0.0, 1.0)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def read_summary(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def group_values(rows: Iterable[Dict[str, str]], group_col: str, metric: str) -> Dict[str, List[float]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        group = row.get(group_col, "")
        value = row.get(metric, "")
        if group == "" or value == "":
            continue
        grouped[group].append(float(value))
    return grouped


def write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--group-col", default="group")
    parser.add_argument("--controls", nargs="*", default=["C1_low_self", "C2_neutral_governance"])
    parser.add_argument("--metrics", nargs="*", default=[])
    parser.add_argument("--out", default="stats_results.csv")
    parser.add_argument("--summary-out", default="group_summary.csv")
    args = parser.parse_args()

    fieldnames, rows = read_summary(args.summary)
    metric_candidates = [c for c in fieldnames if c.startswith("avg_")]
    metrics = args.metrics if args.metrics else metric_candidates

    summary_rows: List[Dict[str, str]] = []
    results_rows: List[Dict[str, str]] = []
    for metric in metrics:
        grouped = group_values(rows, args.group_col, metric)
        for group, values in grouped.items():
            summary_rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "n": str(len(values)),
                    "mean": f"{mean(values):.6f}",
                    "std": f"{std(values):.6f}",
                }
            )
        for control in args.controls:
            if control not in grouped:
                continue
            control_values = grouped[control]
            for group, values in grouped.items():
                if group == control:
                    continue
                t_stat, t_p = welch_t_test(values, control_values)
                d_stat, d_p = ks_statistic(values, control_values)
                results_rows.append(
                    {
                        "metric": metric,
                        "group": group,
                        "control": control,
                        "t_stat": f"{t_stat:.6f}",
                        "t_pvalue": f"{t_p:.6f}",
                        "ks_stat": f"{d_stat:.6f}",
                        "ks_pvalue": f"{d_p:.6f}",
                    }
                )

    write_csv(args.summary_out, summary_rows)
    write_csv(args.out, results_rows)


if __name__ == "__main__":
    main()
