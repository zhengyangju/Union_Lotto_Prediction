# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from fetch_ssq_history import fetch_ssq_history
except Exception as exc:  # pragma: no cover - Streamlit 下仅用于提示
    fetch_ssq_history = None
    FETCH_IMPORT_ERROR = str(exc)
else:
    FETCH_IMPORT_ERROR = ""


DATA_FILE_DEFAULT = "ssq_history.xlsx"
PLOT_DIR = Path("plots")
RED_COLS = ["red_1", "red_2", "red_3", "red_4", "red_5", "red_6"]
BLUE_COL = "blue"
DATE_COL = "draw_date"


def normalize_numeric_str(series: pd.Series, width: int) -> pd.Series:
    # 统一文本格式，避免前导零丢失
    cleaned = series.astype(str).str.replace(r"\.0$", "", regex=True)
    cleaned = cleaned.replace("nan", "")
    return cleaned.str.zfill(width)


@st.cache_data(show_spinner=False)
def load_data(file_path: str) -> pd.DataFrame:
    # 读取 Excel 并做基础清洗
    df = pd.read_excel(file_path, engine="openpyxl")
    df["issue"] = normalize_numeric_str(df["issue"], 5)
    for col in RED_COLS + [BLUE_COL]:
        df[col] = normalize_numeric_str(df[col], 2)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df


def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    # 生成用于计算的数值 DataFrame
    df_num = df.copy()
    for col in RED_COLS + [BLUE_COL]:
        df_num[col] = pd.to_numeric(df_num[col], errors="coerce")
    return df_num


def setup_plot_style() -> None:
    # 绘图统一字体与符号设置
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False


def save_and_show(fig: plt.Figure, name: str) -> Path:
    # 保存图像并返回路径
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIR / f"{name}.jpg"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    return output_path


def render_ball_row(reds: Iterable[str], blue: str) -> None:
    # 号码展示的彩色胶囊样式
    red_html = " ".join([f"<span class='ball red'>{n}</span>" for n in reds])
    blue_html = f"<span class='ball blue'>{blue}</span>"
    st.markdown(f"<div class='ball-row'>{red_html} {blue_html}</div>", unsafe_allow_html=True)


def plot_red_frequency(df_num: pd.DataFrame, tag: str) -> Path:
    setup_plot_style()
    numbers = pd.to_numeric(df_num[RED_COLS].stack(), errors="coerce").dropna().astype(int)
    counts = numbers.value_counts().reindex(range(1, 34), fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(1, 34), counts.values, color="#e74c3c")
    ax.set_title("Red Ball Frequency")
    ax.set_xlabel("Number")
    ax.set_ylabel("Count")
    ax.set_xticks(range(1, 34, 1))
    ax.grid(axis="y", alpha=0.3)
    return save_and_show(fig, f"red_frequency_{tag}")


def plot_blue_frequency(df_num: pd.DataFrame, tag: str) -> Path:
    setup_plot_style()
    numbers = pd.to_numeric(df_num[BLUE_COL], errors="coerce").dropna().astype(int)
    counts = numbers.value_counts().reindex(range(1, 17), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, 17), counts.values, color="#3498db")
    ax.set_title("Blue Ball Frequency")
    ax.set_xlabel("Number")
    ax.set_ylabel("Count")
    ax.set_xticks(range(1, 17, 1))
    ax.grid(axis="y", alpha=0.3)
    return save_and_show(fig, f"blue_frequency_{tag}")


def plot_red_trend(df_num: pd.DataFrame, tag: str) -> Path:
    setup_plot_style()
    df_rev = df_num.iloc[::-1].reset_index(drop=True)
    x = np.repeat(np.arange(len(df_rev)), len(RED_COLS))
    y = df_rev[RED_COLS].astype(int).to_numpy().flatten()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(x, y, s=10, alpha=0.6, color="#2c3e50")
    ax.set_title("Red Ball Trend")
    ax.set_xlabel("Draw Index (Oldest to Newest)")
    ax.set_ylabel("Number")
    ax.set_yticks(range(1, 34, 2))
    ax.grid(axis="y", alpha=0.2)
    return save_and_show(fig, f"red_trend_{tag}")


def plot_blue_trend(df_num: pd.DataFrame, tag: str) -> Path:
    setup_plot_style()
    df_rev = df_num.iloc[::-1].reset_index(drop=True)
    x = np.arange(len(df_rev))
    y = df_rev[BLUE_COL].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, y, color="#1f77b4", linewidth=1.5, marker="o", markersize=3)
    ax.set_title("Blue Ball Trend")
    ax.set_xlabel("Draw Index (Oldest to Newest)")
    ax.set_ylabel("Number")
    ax.set_yticks(range(1, 17, 1))
    ax.grid(axis="y", alpha=0.2)
    return save_and_show(fig, f"blue_trend_{tag}")


def plot_sum_trend(df_num: pd.DataFrame, tag: str) -> Path:
    setup_plot_style()
    df_rev = df_num.iloc[::-1].reset_index(drop=True)
    x = np.arange(len(df_rev))
    red_sum = df_rev[RED_COLS].astype(int).sum(axis=1)
    total_sum = red_sum + df_rev[BLUE_COL].astype(int)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, red_sum, color="#e67e22", label="Red Sum")
    ax.plot(x, total_sum, color="#16a085", label="Total Sum")
    ax.set_title("Sum Trend")
    ax.set_xlabel("Draw Index (Oldest to Newest)")
    ax.set_ylabel("Sum")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    return save_and_show(fig, f"sum_trend_{tag}")


def plot_span_trend(df_num: pd.DataFrame, tag: str) -> Path:
    setup_plot_style()
    df_rev = df_num.iloc[::-1].reset_index(drop=True)
    x = np.arange(len(df_rev))
    span = df_rev[RED_COLS].astype(int).max(axis=1) - df_rev[RED_COLS].astype(int).min(axis=1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, span, color="#8e44ad", linewidth=1.5)
    ax.set_title("Red Ball Span Trend")
    ax.set_xlabel("Draw Index (Oldest to Newest)")
    ax.set_ylabel("Span")
    ax.grid(axis="y", alpha=0.2)
    return save_and_show(fig, f"span_trend_{tag}")


def plot_odd_even_trend(df_num: pd.DataFrame, tag: str) -> Path:
    setup_plot_style()
    df_rev = df_num.iloc[::-1].reset_index(drop=True)
    x = np.arange(len(df_rev))
    red_vals = df_rev[RED_COLS].astype(int)
    odd_count = (red_vals % 2 == 1).sum(axis=1)
    even_count = 6 - odd_count
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, odd_count, color="#c0392b", label="Odd Count")
    ax.plot(x, even_count, color="#2980b9", label="Even Count")
    ax.set_title("Odd vs Even Trend")
    ax.set_xlabel("Draw Index (Oldest to Newest)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    return save_and_show(fig, f"odd_even_trend_{tag}")


def weighted_sample_without_replacement(
    numbers: List[int], weights: List[float], k: int, rng: random.Random
) -> List[int]:
    # 按权重无放回抽样
    pool = numbers[:]
    pool_weights = weights[:]
    selected: List[int] = []
    for _ in range(min(k, len(pool))):
        total = sum(pool_weights)
        if total <= 0:
            pick = rng.choice(pool)
        else:
            pick = rng.choices(pool, weights=pool_weights, k=1)[0]
        idx = pool.index(pick)
        selected.append(pick)
        pool.pop(idx)
        pool_weights.pop(idx)
    return selected


def compute_gaps(df_num: pd.DataFrame, numbers: Iterable[int], cols: List[str]) -> Dict[int, int]:
    # 计算每个号码距最近出现的期数
    gaps = {n: len(df_num) for n in numbers}
    for idx, row in enumerate(df_num[cols].itertuples(index=False)):
        for val in row:
            if pd.isna(val):
                continue
            val = int(val)
            if val in gaps and gaps[val] == len(df_num):
                gaps[val] = idx
    return gaps


def compute_recency_scores(
    df_num: pd.DataFrame, numbers: Iterable[int], cols: List[str], half_life: float
) -> Dict[int, float]:
    # 指数衰减热度评分（越新越高）
    scores = {n: 0.0 for n in numbers}
    decay = math.log(2) / max(1.0, half_life)
    for idx, row in enumerate(df_num[cols].itertuples(index=False)):
        weight = math.exp(-decay * idx)
        for val in row:
            if pd.isna(val):
                continue
            val = int(val)
            if val in scores:
                scores[val] += weight
    return scores


def compute_gap_stats(
    df_num: pd.DataFrame, numbers: Iterable[int], cols: List[str]
) -> Tuple[Dict[int, float], Dict[int, int]]:
    # 计算平均间隔与当前间隔
    positions: Dict[int, List[int]] = {n: [] for n in numbers}
    for idx, row in enumerate(df_num[cols].itertuples(index=False)):
        for val in row:
            if pd.isna(val):
                continue
            val = int(val)
            if val in positions:
                positions[val].append(idx)

    avg_gap: Dict[int, float] = {}
    current_gap: Dict[int, int] = {}
    total_len = len(df_num)
    for num, pos in positions.items():
        if not pos:
            avg_gap[num] = total_len
            current_gap[num] = total_len
            continue
        current_gap[num] = pos[0]
        if len(pos) == 1:
            avg_gap[num] = total_len
        else:
            gaps = [pos[i + 1] - pos[i] for i in range(len(pos) - 1)]
            avg_gap[num] = float(sum(gaps)) / len(gaps)
    return avg_gap, current_gap


def pick_top_by_bucket(
    scores: Dict[int, float], buckets: List[List[int]], per_bucket: int, rng: random.Random
) -> List[int]:
    # 按分区选取高分号码
    selected: List[int] = []
    for bucket in buckets:
        candidates = bucket[:]
        rng.shuffle(candidates)
        candidates.sort(key=lambda n: scores.get(n, 0.0), reverse=True)
        selected.extend(candidates[:per_bucket])
    return selected


def build_ensemble_recommendation(
    method_results: List[Dict[str, object]],
    recency_scores_red: Dict[int, float],
    recency_scores_blue: Dict[int, float],
    rng: random.Random,
) -> Tuple[List[int], int]:
    # 综合投票 + 热度微调 + 分区约束
    all_reds = list(range(1, 34))
    red_votes = {n: 0 for n in all_reds}
    for item in method_results:
        reds = item["reds"]
        for n in reds:
            red_votes[n] += 1

    max_recency = max(recency_scores_red.values()) if recency_scores_red else 1.0
    red_scores = {
        n: red_votes[n] + 0.1 * (recency_scores_red.get(n, 0.0) / max_recency) for n in all_reds
    }

    buckets = [list(range(1, 12)), list(range(12, 23)), list(range(23, 34))]
    selected: List[int] = []
    bucket_counts = [0, 0, 0]

    for idx, bucket in enumerate(buckets):
        candidates = bucket[:]
        rng.shuffle(candidates)
        candidates.sort(key=lambda n: red_scores.get(n, 0.0), reverse=True)
        pick = candidates[0]
        selected.append(pick)
        bucket_counts[idx] += 1

    ranked = sorted(all_reds, key=lambda n: red_scores.get(n, 0.0), reverse=True)
    for n in ranked:
        if n in selected:
            continue
        bucket_idx = 0 if n <= 11 else 1 if n <= 22 else 2
        if bucket_counts[bucket_idx] >= 3:
            continue
        selected.append(n)
        bucket_counts[bucket_idx] += 1
        if len(selected) == 6:
            break

    blue_votes = {n: 0 for n in range(1, 17)}
    for item in method_results:
        blue = int(item["blue"])
        blue_votes[blue] += 1

    max_blue_recency = max(recency_scores_blue.values()) if recency_scores_blue else 1.0
    blue_scores = {
        n: blue_votes[n] + 0.1 * (recency_scores_blue.get(n, 0.0) / max_blue_recency)
        for n in range(1, 17)
    }
    blue_pick = max(blue_scores, key=blue_scores.get)

    return sorted(selected), blue_pick


def get_number_counts(df_num: pd.DataFrame, numbers: Iterable[int], cols: List[str]) -> Dict[int, int]:
    # 统计号码出现次数
    values = pd.to_numeric(df_num[cols].stack(), errors="coerce").dropna().astype(int)
    counts = values.value_counts().to_dict()
    return {n: counts.get(n, 0) for n in numbers}


def bucket_index(num: int) -> int:
    # 将号码映射到三个区间
    if num <= 11:
        return 0
    if num <= 22:
        return 1
    return 2


def count_buckets(nums: Iterable[int]) -> List[int]:
    # 统计区间分布
    counts = [0, 0, 0]
    for n in nums:
        counts[bucket_index(int(n))] += 1
    return counts


def compute_bucket_target(df_num: pd.DataFrame) -> List[int]:
    # 计算区间目标分布并校正为 6 个号码
    red_vals = df_num[RED_COLS].astype(int)
    bucket_counts: List[List[int]] = []
    for row in red_vals.itertuples(index=False):
        bucket_counts.append(count_buckets(row))
    if not bucket_counts:
        return [2, 2, 2]
    mean_counts = np.mean(bucket_counts, axis=0)
    target = [int(round(val)) for val in mean_counts]
    while sum(target) > 6:
        idx = target.index(max(target))
        target[idx] -= 1
    while sum(target) < 6:
        idx = target.index(min(target))
        target[idx] += 1
    return target


def compute_red_stats(df_num: pd.DataFrame) -> Dict[str, float | List[int]]:
    # 统计红球结构特征
    red_vals = df_num[RED_COLS].astype(int)
    sum_series = red_vals.sum(axis=1)
    span_series = red_vals.max(axis=1) - red_vals.min(axis=1)
    odd_series = (red_vals % 2 == 1).sum(axis=1)
    return {
        "sum_mean": float(sum_series.mean()),
        "sum_std": float(sum_series.std(ddof=0) or 1.0),
        "span_mean": float(span_series.mean()),
        "span_std": float(span_series.std(ddof=0) or 1.0),
        "odd_mean": float(odd_series.mean()),
        "bucket_target": compute_bucket_target(df_num),
    }


def sample_red_numbers(rng: random.Random, weights: Dict[int, float] | None = None) -> List[int]:
    # 根据权重抽取红球
    if weights:
        numbers = list(weights.keys())
        w = [weights[n] for n in numbers]
        return sorted(weighted_sample_without_replacement(numbers, w, 6, rng))
    return sorted(rng.sample(range(1, 34), 6))


def score_candidate_set(
    nums: List[int],
    target_sum: float,
    target_span: float,
    target_odd: float,
    bucket_target: List[int],
    sum_scale: float,
    span_scale: float,
) -> float:
    # 候选组合评分（越小越好）
    nums_sorted = sorted(nums)
    sum_val = sum(nums_sorted)
    span_val = nums_sorted[-1] - nums_sorted[0]
    odd_count = sum(1 for n in nums_sorted if n % 2 == 1)
    bucket_counts = count_buckets(nums_sorted)
    consecutive = sum(1 for i in range(5) if nums_sorted[i + 1] - nums_sorted[i] == 1)

    sum_score = abs(sum_val - target_sum) / max(1.0, sum_scale)
    span_score = abs(span_val - target_span) / max(1.0, span_scale)
    odd_score = abs(odd_count - target_odd)
    bucket_score = sum(abs(bucket_counts[i] - bucket_target[i]) for i in range(3))
    consecutive_score = max(0, consecutive - 1)

    return sum_score + span_score + 0.6 * odd_score + 0.4 * bucket_score + 0.2 * consecutive_score


def random_search_red_set(
    rng: random.Random,
    iterations: int,
    target_sum: float,
    target_span: float,
    target_odd: float,
    bucket_target: List[int],
    sum_scale: float,
    span_scale: float,
    weights: Dict[int, float] | None = None,
) -> List[int]:
    # 随机搜索最优红球组合
    best = sample_red_numbers(rng, weights)
    best_score = score_candidate_set(best, target_sum, target_span, target_odd, bucket_target, sum_scale, span_scale)
    for _ in range(iterations):
        candidate = sample_red_numbers(rng, weights)
        score = score_candidate_set(
            candidate, target_sum, target_span, target_odd, bucket_target, sum_scale, span_scale
        )
        if score < best_score:
            best = candidate
            best_score = score
    return best


def pick_blue_by_target(target: float) -> int:
    # 选择最接近目标值的蓝球
    return min(range(1, 17), key=lambda n: abs(n - target))


def compute_markov_scores(
    df_num: pd.DataFrame, numbers: Iterable[int], cols: List[str]
) -> Dict[int, float]:
    # 计算一阶马尔可夫转移概率
    draws = []
    for row in df_num[cols].itertuples(index=False):
        draws.append({int(v) for v in row if not pd.isna(v)})
    if len(draws) < 2:
        return {n: 0.0 for n in numbers}

    draws = list(reversed(draws))
    counts = {n: [[0, 0], [0, 0]] for n in numbers}
    for idx in range(len(draws) - 1):
        prev_set = draws[idx]
        next_set = draws[idx + 1]
        for n in numbers:
            prev_state = 1 if n in prev_set else 0
            next_state = 1 if n in next_set else 0
            counts[n][prev_state][next_state] += 1

    latest_set = draws[-1]
    scores = {}
    for n in numbers:
        prev_state = 1 if n in latest_set else 0
        stay = counts[n][prev_state][1] + 1
        leave = counts[n][prev_state][0] + 1
        scores[n] = stay / (stay + leave)
    return scores


def compute_pmi_matrix(df_num: pd.DataFrame, numbers: List[int], cols: List[str]) -> Dict[int, Dict[int, float]]:
    # 计算号码互信息（PMI）
    draws = []
    for row in df_num[cols].itertuples(index=False):
        draws.append(sorted({int(v) for v in row if not pd.isna(v)}))
    total = max(1, len(draws))

    count = {n: 0 for n in numbers}
    co_counts: Dict[Tuple[int, int], int] = {}
    for draw in draws:
        for n in draw:
            count[n] += 1
        for i in range(len(draw)):
            for j in range(i + 1, len(draw)):
                pair = (draw[i], draw[j])
                co_counts[pair] = co_counts.get(pair, 0) + 1

    pmi = {n: {} for n in numbers}
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            a, b = numbers[i], numbers[j]
            p_a = (count[a] + 1) / (total + 2)
            p_b = (count[b] + 1) / (total + 2)
            p_ab = (co_counts.get((a, b), 0) + 1) / (total + 2)
            value = math.log(p_ab / (p_a * p_b))
            pmi[a][b] = value
            pmi[b][a] = value
    return pmi


def pick_low_pmi_set(
    pmi: Dict[int, Dict[int, float]], numbers: List[int], rng: random.Random
) -> List[int]:
    # 贪心选择低互信息组合
    mean_pmi = {
        n: float(np.mean([pmi[n][m] for m in numbers if m != n])) for n in numbers
    }
    start = min(mean_pmi, key=mean_pmi.get)
    selected = [start]
    bucket_counts = count_buckets(selected)

    while len(selected) < 6:
        best = None
        best_score = float("inf")
        for n in numbers:
            if n in selected:
                continue
            bucket_idx = bucket_index(n)
            if bucket_counts[bucket_idx] >= 3:
                continue
            score = float(np.mean([pmi[n][s] for s in selected]))
            if score < best_score:
                best_score = score
                best = n
        if best is None:
            break
        selected.append(best)
        bucket_counts[bucket_index(best)] += 1

    if len(selected) < 6:
        remaining = [n for n in numbers if n not in selected]
        rng.shuffle(remaining)
        remaining.sort(key=lambda n: mean_pmi.get(n, 0.0))
        for n in remaining:
            if len(selected) == 6:
                break
            bucket_idx = bucket_index(n)
            if bucket_counts[bucket_idx] >= 3:
                continue
            selected.append(n)
            bucket_counts[bucket_idx] += 1

    return sorted(selected)


def poisson_score(k: int, lam: float) -> float:
    # 泊松分布概率（对数形式转回）
    if lam <= 0:
        return 0.0
    log_p = -lam + k * math.log(lam) - math.lgamma(k + 1)
    return math.exp(log_p)


def predict_method_entropy(df_num: pd.DataFrame, rng: random.Random) -> Tuple[List[int], int]:
    # 创意方法一：分段熵平衡（冷热反向权重）
    all_numbers = list(range(1, 34))
    freq = pd.to_numeric(df_num[RED_COLS].stack(), errors="coerce").dropna().astype(int)
    freq_counts = freq.value_counts().to_dict()
    weights = {n: 1.0 / (freq_counts.get(n, 0) + 1) for n in all_numbers}

    buckets = [list(range(1, 12)), list(range(12, 23)), list(range(23, 34))]
    chosen: List[int] = []
    for bucket in buckets:
        bucket_weights = [weights[n] for n in bucket]
        chosen.extend(weighted_sample_without_replacement(bucket, bucket_weights, 2, rng))

    blue_freq = pd.to_numeric(df_num[BLUE_COL], errors="coerce").dropna().astype(int)
    blue_counts = blue_freq.value_counts().to_dict()
    blue_weights = [1.0 / (blue_counts.get(n, 0) + 1) for n in range(1, 17)]
    blue_pick = weighted_sample_without_replacement(list(range(1, 17)), blue_weights, 1, rng)[0]
    return sorted(chosen), blue_pick


def predict_method_gap_wave(df_num: pd.DataFrame, rng: random.Random) -> Tuple[List[int], int]:
    # 创意方法二：间隔波动（偏好中等空档）
    numbers = list(range(1, 34))
    gaps = compute_gaps(df_num, numbers, RED_COLS)
    gap_values = list(gaps.values())
    median_gap = sorted(gap_values)[len(gap_values) // 2]
    sigma = max(1.0, len(df_num) / 6.0)
    weights = []
    for n in numbers:
        score = math.exp(-((gaps[n] - median_gap) ** 2) / (2 * sigma**2))
        weights.append(score)
    chosen = weighted_sample_without_replacement(numbers, weights, 6, rng)

    blue_numbers = list(range(1, 17))
    blue_gaps = compute_gaps(df_num, blue_numbers, [BLUE_COL])
    blue_values = list(blue_gaps.values())
    blue_median = sorted(blue_values)[len(blue_values) // 2]
    blue_sigma = max(1.0, len(df_num) / 6.0)
    blue_weights = [
        math.exp(-((blue_gaps[n] - blue_median) ** 2) / (2 * blue_sigma**2)) for n in blue_numbers
    ]
    blue_pick = weighted_sample_without_replacement(blue_numbers, blue_weights, 1, rng)[0]
    return sorted(chosen), blue_pick


def predict_method_anti_cluster(df_num: pd.DataFrame) -> Tuple[List[int], int]:
    # 创意方法三：反聚类协同网络（弱相关组合）
    size = 34
    co = [[0 for _ in range(size)] for _ in range(size)]
    for row in df_num[RED_COLS].itertuples(index=False):
        nums = sorted({int(v) for v in row if not pd.isna(v)})
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                co[nums[i]][nums[j]] += 1
                co[nums[j]][nums[i]] += 1

    strength = {n: sum(co[n]) for n in range(1, 34)}
    selected = [max(strength, key=strength.get)]
    while len(selected) < 6:
        candidates = [n for n in range(1, 34) if n not in selected]
        scores = {n: sum(co[n][s] for s in selected) for n in candidates}
        best_score = min(scores.values())
        best_candidates = [n for n, s in scores.items() if s == best_score]
        best_candidates.sort(key=lambda n: strength[n], reverse=True)
        selected.append(best_candidates[0])

    blue_freq = pd.to_numeric(df_num[BLUE_COL], errors="coerce").dropna().astype(int)
    blue_counts = blue_freq.value_counts()
    blue_pick = int(blue_counts.idxmax()) if not blue_counts.empty else 1
    return sorted(selected), blue_pick


def predict_method_recency_hot(
    df_num: pd.DataFrame, recency_scores_red: Dict[int, float], recency_scores_blue: Dict[int, float], rng: random.Random
) -> Tuple[List[int], int]:
    # 创意方法四：指数记忆热度（近期高权重）
    buckets = [list(range(1, 12)), list(range(12, 23)), list(range(23, 34))]
    chosen = pick_top_by_bucket(recency_scores_red, buckets, 2, rng)

    blue_candidates = list(range(1, 17))
    rng.shuffle(blue_candidates)
    blue_candidates.sort(key=lambda n: recency_scores_blue.get(n, 0.0), reverse=True)
    blue_pick = blue_candidates[0]
    return sorted(chosen), blue_pick


def predict_method_cycle_reversion(
    df_num: pd.DataFrame, rng: random.Random
) -> Tuple[List[int], int]:
    # 创意方法五：周期回归（当前间隔接近历史均值）
    numbers = list(range(1, 34))
    avg_gap, current_gap = compute_gap_stats(df_num, numbers, RED_COLS)
    scores = {n: 1.0 / (abs(current_gap[n] - avg_gap[n]) + 1.0) for n in numbers}

    buckets = [list(range(1, 12)), list(range(12, 23)), list(range(23, 34))]
    chosen = pick_top_by_bucket(scores, buckets, 2, rng)

    blue_numbers = list(range(1, 17))
    blue_avg, blue_current = compute_gap_stats(df_num, blue_numbers, [BLUE_COL])
    blue_scores = {
        n: 1.0 / (abs(blue_current[n] - blue_avg[n]) + 1.0) for n in blue_numbers
    }
    blue_candidates = blue_numbers[:]
    rng.shuffle(blue_candidates)
    blue_candidates.sort(key=lambda n: blue_scores.get(n, 0.0), reverse=True)
    blue_pick = blue_candidates[0]
    return sorted(chosen), blue_pick


def predict_method_mirror(
    df_num: pd.DataFrame,
    latest_row: pd.Series,
    recency_scores_red: Dict[int, float],
    rng: random.Random,
) -> Tuple[List[int], int]:
    # 创意方法六：镜像映射（围绕中点对称并做轻微扰动）
    latest_reds = [int(latest_row[col]) for col in RED_COLS]
    mirrored = [34 - n for n in latest_reds]
    chosen: List[int] = []

    for val in mirrored:
        candidate = val
        if candidate in chosen:
            options = [candidate - 1, candidate + 1]
            options = [o for o in options if 1 <= o <= 33 and o not in chosen]
            if options:
                options.sort(key=lambda n: recency_scores_red.get(n, 0.0))
                candidate = options[0]
            else:
                candidate = None
        if candidate is not None and candidate not in chosen:
            chosen.append(candidate)

    if len(chosen) < 6:
        cold_numbers = list(range(1, 34))
        rng.shuffle(cold_numbers)
        cold_numbers.sort(key=lambda n: recency_scores_red.get(n, 0.0))
        for n in cold_numbers:
            if n not in chosen:
                chosen.append(n)
            if len(chosen) == 6:
                break

    latest_blue = int(latest_row[BLUE_COL])
    blue_pick = 17 - latest_blue
    if blue_pick < 1 or blue_pick > 16:
        blue_pick = ((latest_blue + 7) % 16) + 1
    return sorted(chosen), blue_pick


def predict_method_markov(df_num: pd.DataFrame, rng: random.Random) -> Tuple[List[int], int]:
    # 方法七：马尔可夫转移
    red_scores = compute_markov_scores(df_num, range(1, 34), RED_COLS)
    buckets = [list(range(1, 12)), list(range(12, 23)), list(range(23, 34))]
    reds = pick_top_by_bucket(red_scores, buckets, 2, rng)

    blue_scores = compute_markov_scores(df_num, range(1, 17), [BLUE_COL])
    blue_candidates = list(range(1, 17))
    rng.shuffle(blue_candidates)
    blue_candidates.sort(key=lambda n: blue_scores.get(n, 0.0), reverse=True)
    return sorted(reds), blue_candidates[0]


def predict_method_bayesian(df_num: pd.DataFrame, rng: random.Random) -> Tuple[List[int], int]:
    # 方法八：贝叶斯更新
    draws = len(df_num)
    alpha, beta = 1.0, 1.0
    red_counts = get_number_counts(df_num, range(1, 34), RED_COLS)
    red_scores = {n: (alpha + red_counts[n]) / (alpha + beta + draws) for n in range(1, 34)}
    buckets = [list(range(1, 12)), list(range(12, 23)), list(range(23, 34))]
    reds = pick_top_by_bucket(red_scores, buckets, 2, rng)

    blue_counts = get_number_counts(df_num, range(1, 17), [BLUE_COL])
    blue_scores = {n: (alpha + blue_counts[n]) / (alpha + beta + draws) for n in range(1, 17)}
    blue_candidates = list(range(1, 17))
    rng.shuffle(blue_candidates)
    blue_candidates.sort(key=lambda n: blue_scores.get(n, 0.0), reverse=True)
    return sorted(reds), blue_candidates[0]


def predict_method_multinomial_poisson(df_num: pd.DataFrame, rng: random.Random) -> Tuple[List[int], int]:
    # 方法九：多项分布/泊松稳定度
    draws = len(df_num)
    red_counts = get_number_counts(df_num, range(1, 34), RED_COLS)
    expected_red = draws * 6 / 33
    red_scores = {n: poisson_score(red_counts[n], expected_red) for n in range(1, 34)}
    buckets = [list(range(1, 12)), list(range(12, 23)), list(range(23, 34))]
    reds = pick_top_by_bucket(red_scores, buckets, 2, rng)

    blue_counts = get_number_counts(df_num, range(1, 17), [BLUE_COL])
    expected_blue = draws / 16
    blue_scores = {n: poisson_score(blue_counts[n], expected_blue) for n in range(1, 17)}
    blue_candidates = list(range(1, 17))
    rng.shuffle(blue_candidates)
    blue_candidates.sort(key=lambda n: blue_scores.get(n, 0.0), reverse=True)
    return sorted(reds), blue_candidates[0]


def predict_method_time_series(
    df_num: pd.DataFrame, red_stats: Dict[str, float | List[int]], rng: random.Random
) -> Tuple[List[int], int]:
    # 方法十：时间序列分解（趋势预测）
    red_vals = df_num[RED_COLS].astype(int)
    sum_series = red_vals.sum(axis=1).iloc[::-1].reset_index(drop=True)
    window = min(30, len(sum_series))
    trend = sum_series.rolling(window=window, min_periods=max(5, window // 2)).mean()
    target_sum = float(trend.iloc[-1]) if not trend.empty else float(red_stats["sum_mean"])

    reds = random_search_red_set(
        rng=rng,
        iterations=2000,
        target_sum=target_sum,
        target_span=float(red_stats["span_mean"]),
        target_odd=float(red_stats["odd_mean"]),
        bucket_target=red_stats["bucket_target"],
        sum_scale=float(red_stats["sum_std"]),
        span_scale=float(red_stats["span_std"]),
    )

    blue_series = df_num[BLUE_COL].astype(int).iloc[::-1].reset_index(drop=True)
    blue_window = min(20, len(blue_series))
    blue_trend = blue_series.rolling(window=blue_window, min_periods=max(3, blue_window // 2)).mean()
    blue_target = float(blue_trend.iloc[-1]) if not blue_trend.empty else float(blue_series.mean())
    blue_pick = pick_blue_by_target(blue_target)
    return sorted(reds), blue_pick


def predict_method_mutual_info(df_num: pd.DataFrame, rng: random.Random) -> Tuple[List[int], int]:
    # 方法十一：互信息网络（弱相关组合）
    numbers = list(range(1, 34))
    pmi = compute_pmi_matrix(df_num, numbers, RED_COLS)
    reds = pick_low_pmi_set(pmi, numbers, rng)

    blue_counts = get_number_counts(df_num, range(1, 17), [BLUE_COL])
    blue_candidates = list(range(1, 17))
    rng.shuffle(blue_candidates)
    blue_candidates.sort(key=lambda n: blue_counts.get(n, 0), reverse=True)
    return sorted(reds), blue_candidates[0]


def predict_method_combo_opt(
    df_num: pd.DataFrame, red_stats: Dict[str, float | List[int]], rng: random.Random
) -> Tuple[List[int], int]:
    # 方法十二：组合优化（多目标约束）
    reds = random_search_red_set(
        rng=rng,
        iterations=3000,
        target_sum=float(red_stats["sum_mean"]),
        target_span=float(red_stats["span_mean"]),
        target_odd=float(red_stats["odd_mean"]),
        bucket_target=red_stats["bucket_target"],
        sum_scale=float(red_stats["sum_std"]),
        span_scale=float(red_stats["span_std"]),
    )
    blue_mean = float(pd.to_numeric(df_num[BLUE_COL], errors="coerce").dropna().mean())
    blue_pick = pick_blue_by_target(blue_mean)
    return sorted(reds), blue_pick


def predict_method_monte_carlo(df_num: pd.DataFrame, rng: random.Random) -> Tuple[List[int], int]:
    # 方法十三：Bootstrap/Monte Carlo 模拟
    numbers = list(range(1, 34))
    blue_numbers = list(range(1, 17))
    red_counts = get_number_counts(df_num, numbers, RED_COLS)
    blue_counts = get_number_counts(df_num, blue_numbers, [BLUE_COL])

    red_weights = [red_counts[n] + 1 for n in numbers]
    blue_weights = [blue_counts[n] + 1 for n in blue_numbers]
    sim_red = {n: 0 for n in numbers}
    sim_blue = {n: 0 for n in blue_numbers}

    for _ in range(2000):
        reds = weighted_sample_without_replacement(numbers, red_weights, 6, rng)
        for n in reds:
            sim_red[n] += 1
        blue = rng.choices(blue_numbers, weights=blue_weights, k=1)[0]
        sim_blue[blue] += 1

    buckets = [list(range(1, 12)), list(range(12, 23)), list(range(23, 34))]
    reds = pick_top_by_bucket(sim_red, buckets, 2, rng)
    blue_pick = max(sim_blue, key=sim_blue.get)
    return sorted(reds), blue_pick


def predict_method_volatility_reversion(
    df_num: pd.DataFrame, red_stats: Dict[str, float | List[int]], rng: random.Random
) -> Tuple[List[int], int]:
    # 方法十四：波动回归（和值均值回归）
    red_vals = df_num[RED_COLS].astype(int)
    sum_series = red_vals.sum(axis=1).iloc[::-1].reset_index(drop=True)
    mean_val = float(sum_series.mean())
    std_val = float(sum_series.std(ddof=0) or 1.0)
    last_sum = float(sum_series.iloc[-1]) if not sum_series.empty else mean_val
    shift = 0.3 * std_val
    target_sum = mean_val - shift if last_sum > mean_val else mean_val + shift

    reds = random_search_red_set(
        rng=rng,
        iterations=2000,
        target_sum=target_sum,
        target_span=float(red_stats["span_mean"]),
        target_odd=float(red_stats["odd_mean"]),
        bucket_target=red_stats["bucket_target"],
        sum_scale=float(red_stats["sum_std"]),
        span_scale=float(red_stats["span_std"]),
    )
    blue_mean = float(pd.to_numeric(df_num[BLUE_COL], errors="coerce").dropna().mean())
    blue_pick = pick_blue_by_target(blue_mean)
    return sorted(reds), blue_pick


def predict_method_phase_space(
    df_num: pd.DataFrame, red_stats: Dict[str, float | List[int]], rng: random.Random
) -> Tuple[List[int], int]:
    # 方法十五：复杂系统相空间类比
    red_vals = df_num[RED_COLS].astype(int)
    sum_series = red_vals.sum(axis=1).iloc[::-1].reset_index(drop=True)
    embed_dim = 3
    target_sum = float(red_stats["sum_mean"])
    if len(sum_series) >= embed_dim + 2:
        target_vec = sum_series.iloc[-embed_dim:].to_numpy(dtype=float)
        candidates: List[Tuple[float, float]] = []
        for i in range(embed_dim - 1, len(sum_series) - 1):
            vec = sum_series.iloc[i - embed_dim + 1 : i + 1].to_numpy(dtype=float)
            dist = float(np.linalg.norm(vec - target_vec))
            next_sum = float(sum_series.iloc[i + 1])
            candidates.append((dist, next_sum))
        candidates.sort(key=lambda x: x[0])
        top = [c[1] for c in candidates[: min(8, len(candidates))]]
        if top:
            target_sum = float(np.median(top))

    reds = random_search_red_set(
        rng=rng,
        iterations=2000,
        target_sum=target_sum,
        target_span=float(red_stats["span_mean"]),
        target_odd=float(red_stats["odd_mean"]),
        bucket_target=red_stats["bucket_target"],
        sum_scale=float(red_stats["sum_std"]),
        span_scale=float(red_stats["span_std"]),
    )
    blue_series = df_num[BLUE_COL].astype(int).iloc[::-1].reset_index(drop=True)
    blue_target = float(blue_series.tail(10).mean()) if not blue_series.empty else 8.0
    blue_pick = pick_blue_by_target(blue_target)
    return sorted(reds), blue_pick


def format_ticket(reds: List[int], blue: int) -> str:
    # 输出标准号码格式
    red_text = " ".join(f"{n:02d}" for n in reds)
    return f"{red_text} + {blue:02d}"


def main() -> None:
    st.set_page_config(page_title="SSQ Visual Lab", layout="wide")

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.8rem; }
        .ball-row { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
        .ball { display: inline-block; padding: 6px 10px; border-radius: 999px; font-weight: 600; color: #fff; }
        .ball.red { background: #e74c3c; }
        .ball.blue { background: #3498db; }
        div[data-testid="stMetric"] { background: #ffffff; padding: 12px 16px; border-radius: 10px; border: 1px solid #f0f0f0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("双色球数据分析与趋势实验室")
    st.caption("提示：预测结果仅供娱乐与数学实验展示，不构成任何投注建议。")

    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.info(f"当前时间：{now_text}")

    with st.sidebar:
        st.header("控制面板")
        data_file = st.text_input("数据文件路径", value=DATA_FILE_DEFAULT)
        data_path = Path(data_file)

        st.subheader("历史数据下载")
        limit = st.number_input("下载期数上限", min_value=100, max_value=10000, value=5000, step=100)
        if st.button("下载/更新历史数据"):
            if fetch_ssq_history is None:
                st.error(f"无法调用抓取模块：{FETCH_IMPORT_ERROR}")
            else:
                with st.spinner("正在下载并保存..."):
                    df_new = fetch_ssq_history(int(limit))
                    df_new.to_excel(data_path, index=False, engine="openpyxl")
                    st.cache_data.clear()
                st.success(f"已保存：{data_path.resolve()}")

        st.subheader("分析期数")
        mode = st.radio("选择范围", ["最近100期", "自定义期数"], index=0)
        custom_n = st.number_input("自定义期数", min_value=20, max_value=10000, value=300, step=10)

    if not data_path.exists():
        st.warning("未找到数据文件，请先下载历史数据。")
        return

    df = load_data(str(data_path))
    if df.empty:
        st.warning("数据为空，请检查 Excel 文件内容。")
        return

    df = df.sort_values("issue", ascending=False).reset_index(drop=True)
    n_periods = min(100, len(df)) if mode == "最近100期" else min(int(custom_n), len(df))

    df_recent = df.head(n_periods).copy()
    df_recent_num = to_numeric_df(df_recent)

    latest = df.iloc[0]
    latest_num = df_recent_num.iloc[0]
    red_latest = [latest[col] for col in RED_COLS]
    blue_latest = latest[BLUE_COL]

    latest_issue_seed = int(str(latest["issue"]).lstrip("0") or "0")
    latest_date_seed = int(latest[DATE_COL].strftime("%Y%m%d")) if pd.notna(latest[DATE_COL]) else 0
    base_seed = latest_issue_seed + latest_date_seed
    recency_scores_red = compute_recency_scores(df_recent_num, range(1, 34), RED_COLS, half_life=60)
    recency_scores_blue = compute_recency_scores(df_recent_num, range(1, 17), [BLUE_COL], half_life=40)
    red_stats = compute_red_stats(df_recent_num)

    metric_cols = st.columns(4)
    metric_cols[0].metric("最新期号", latest["issue"])
    metric_cols[1].metric("最新日期", latest[DATE_COL].strftime("%Y-%m-%d") if pd.notna(latest[DATE_COL]) else "N/A")
    metric_cols[2].metric("总期数", len(df))
    metric_cols[3].metric("当前分析期数", n_periods)

    st.subheader("最新一期号码")
    render_ball_row(red_latest, blue_latest)

    tab_freq, tab_trend, tab_structure, tab_predict, tab_data = st.tabs(
        ["频次分布", "号码走势", "结构趋势", "创意预测", "原始数据"]
    )

    with tab_freq:
        st.markdown("**频次统计（红球与蓝球）**")
        tag = f"freq_{n_periods}"
        path_red = plot_red_frequency(df_recent_num, tag)
        path_blue = plot_blue_frequency(df_recent_num, tag)
        st.caption(f"图片已保存：{path_red}")
        st.caption(f"图片已保存：{path_blue}")

    with tab_trend:
        st.markdown("**走势散点与序列趋势**")
        tag = f"trend_{n_periods}"
        path_red = plot_red_trend(df_recent_num, tag)
        path_blue = plot_blue_trend(df_recent_num, tag)
        st.caption(f"图片已保存：{path_red}")
        st.caption(f"图片已保存：{path_blue}")

    with tab_structure:
        st.markdown("**和值、跨度与奇偶结构**")
        tag = f"struct_{n_periods}"
        path_sum = plot_sum_trend(df_recent_num, tag)
        path_span = plot_span_trend(df_recent_num, tag)
        path_oe = plot_odd_even_trend(df_recent_num, tag)
        st.caption(f"图片已保存：{path_sum}")
        st.caption(f"图片已保存：{path_span}")
        st.caption(f"图片已保存：{path_oe}")

    with tab_predict:
        st.markdown("**数学创意预测（仅供娱乐）**")
        rng_entropy = random.Random(base_seed + 11)
        rng_gap = random.Random(base_seed + 23)
        rng_markov = random.Random(base_seed + 31)
        rng_bayes = random.Random(base_seed + 37)
        rng_poisson = random.Random(base_seed + 41)
        rng_time = random.Random(base_seed + 47)
        rng_mi = random.Random(base_seed + 53)
        rng_combo = random.Random(base_seed + 59)
        rng_mc = random.Random(base_seed + 61)
        rng_vol = random.Random(base_seed + 67)
        rng_phase = random.Random(base_seed + 71)
        rng_hot = random.Random(base_seed + 73)
        rng_cycle = random.Random(base_seed + 79)
        rng_mirror = random.Random(base_seed + 83)
        rng_reco = random.Random(base_seed + 97)

        method_results: List[Dict[str, object]] = []

        reds_a, blue_a = predict_method_entropy(df_recent_num, rng_entropy)
        method_results.append(
            {
                "name": "方法一：分段熵平衡（冷热反向权重）",
                "desc": "思路：把红球分为三个区间，各区间以冷号为主进行无放回抽样，形成分布更均匀的组合。",
                "reds": reds_a,
                "blue": blue_a,
            }
        )

        reds_b, blue_b = predict_method_gap_wave(df_recent_num, rng_gap)
        method_results.append(
            {
                "name": "方法二：间隔波动（偏好中等空档）",
                "desc": "思路：计算号码最近出现间隔，偏向选择空档处于中位数附近的号码。",
                "reds": reds_b,
                "blue": blue_b,
            }
        )

        reds_c, blue_c = predict_method_anti_cluster(df_recent_num)
        method_results.append(
            {
                "name": "方法三：反聚类协同网络（弱相关组合）",
                "desc": "思路：基于共现矩阵选取相互共现次数较少的号码，强调组合多样性。",
                "reds": reds_c,
                "blue": blue_c,
            }
        )

        reds_d, blue_d = predict_method_markov(df_recent_num, rng_markov)
        method_results.append(
            {
                "name": "方法四：马尔可夫转移（状态概率）",
                "desc": "思路：基于上一期是否出现构造转移概率，估计下一期出现倾向。",
                "reds": reds_d,
                "blue": blue_d,
            }
        )

        reds_e, blue_e = predict_method_bayesian(df_recent_num, rng_bayes)
        method_results.append(
            {
                "name": "方法五：贝叶斯更新（后验均值）",
                "desc": "思路：以先验为基底，用出现频次更新后验概率。",
                "reds": reds_e,
                "blue": blue_e,
            }
        )

        reds_f, blue_f = predict_method_multinomial_poisson(df_recent_num, rng_poisson)
        method_results.append(
            {
                "name": "方法六：多项/泊松稳定度",
                "desc": "思路：根据泊松模型评估出现次数与期望值的贴合度。",
                "reds": reds_f,
                "blue": blue_f,
            }
        )

        reds_g, blue_g = predict_method_time_series(df_recent_num, red_stats, rng_time)
        method_results.append(
            {
                "name": "方法七：时间序列趋势（和值预测）",
                "desc": "思路：用滚动趋势估计下一期和值并匹配组合。",
                "reds": reds_g,
                "blue": blue_g,
            }
        )

        reds_h, blue_h = predict_method_mutual_info(df_recent_num, rng_mi)
        method_results.append(
            {
                "name": "方法八：互信息网络（弱关联）",
                "desc": "思路：优先选择互信息较低的号码组合，降低相关性。",
                "reds": reds_h,
                "blue": blue_h,
            }
        )

        reds_i, blue_i = predict_method_combo_opt(df_recent_num, red_stats, rng_combo)
        method_results.append(
            {
                "name": "方法九：组合优化（多目标约束）",
                "desc": "思路：同时约束和值、跨度、奇偶与区间分布，搜索最优组合。",
                "reds": reds_i,
                "blue": blue_i,
            }
        )

        reds_j, blue_j = predict_method_monte_carlo(df_recent_num, rng_mc)
        method_results.append(
            {
                "name": "方法十：Bootstrap/Monte Carlo 模拟",
                "desc": "思路：用概率模型进行多次模拟，统计出现频率并择优。",
                "reds": reds_j,
                "blue": blue_j,
            }
        )

        reds_k, blue_k = predict_method_volatility_reversion(df_recent_num, red_stats, rng_vol)
        method_results.append(
            {
                "name": "方法十一：波动回归（和值均值回归）",
                "desc": "思路：基于最近和值偏离程度进行均值回归预测。",
                "reds": reds_k,
                "blue": blue_k,
            }
        )

        reds_l, blue_l = predict_method_phase_space(df_recent_num, red_stats, rng_phase)
        method_results.append(
            {
                "name": "方法十二：复杂系统相空间类比",
                "desc": "思路：以相空间相似序列的后续走势估计目标和值。",
                "reds": reds_l,
                "blue": blue_l,
            }
        )

        reds_m, blue_m = predict_method_recency_hot(
            df_recent_num, recency_scores_red, recency_scores_blue, rng_hot
        )
        method_results.append(
            {
                "name": "方法十三：指数记忆热度（近期高权重）",
                "desc": "思路：对近期开奖给予更高权重，倾向选取近期活跃号码。",
                "reds": reds_m,
                "blue": blue_m,
            }
        )

        reds_n, blue_n = predict_method_cycle_reversion(df_recent_num, rng_cycle)
        method_results.append(
            {
                "name": "方法十四：周期回归（间隔接近均值）",
                "desc": "思路：选择当前间隔接近历史均值的号码，体现周期回归假设。",
                "reds": reds_n,
                "blue": blue_n,
            }
        )

        reds_o, blue_o = predict_method_mirror(
            df_recent_num, latest_num, recency_scores_red, rng_mirror
        )
        method_results.append(
            {
                "name": "方法十五：镜像映射（对称扰动）",
                "desc": "思路：对最近一期号码做中点对称映射，并用冷号轻微扰动补齐。",
                "reds": reds_o,
                "blue": blue_o,
            }
        )

        recommend_reds, recommend_blue = build_ensemble_recommendation(
            method_results, recency_scores_red, recency_scores_blue, rng_reco
        )

        st.markdown("**综合推荐号码**")
        render_ball_row([f"{n:02d}" for n in recommend_reds], f"{recommend_blue:02d}")
        st.code(format_ticket(recommend_reds, recommend_blue))
        st.caption("综合规则：多方法投票 + 近期热度微调 + 分区约束。")

        for idx, item in enumerate(method_results):
            expanded = idx == 0
            with st.expander(item["name"], expanded=expanded):
                st.write(item["desc"])
                st.code(format_ticket(item["reds"], int(item["blue"])))

    with tab_data:
        st.markdown("**最近数据预览**")
        st.dataframe(df_recent, use_container_width=True, height=420)


if __name__ == "__main__":
    main()
