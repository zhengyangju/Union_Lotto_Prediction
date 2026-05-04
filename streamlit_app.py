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
except Exception as exc:  
    fetch_ssq_history = None
    FETCH_IMPORT_ERROR = str(exc)
else:
    FETCH_IMPORT_ERROR = ""

try:
    from fetch_sd_history import fetch_sd_history
except Exception as exc:  # pragma: no cover - Streamlit下仅用于提示
    fetch_sd_history = None
    FETCH_SD_IMPORT_ERROR = str(exc)
else:
    FETCH_SD_IMPORT_ERROR = ""

try:
    from fetch_dlt_history import fetch_dlt_history
except Exception as exc:  # pragma: no cover - Streamlit 下仅用于提示
    fetch_dlt_history = None
    FETCH_DLT_IMPORT_ERROR = str(exc)
else:
    FETCH_DLT_IMPORT_ERROR = ""


DATA_FILE_DEFAULT = "ssq_history.xlsx"
SD_FILE_DEFAULT = "sd_history.xlsx"
DLT_FILE_DEFAULT = "dlt_history.xlsx"
PLOT_DIR = Path("plots")
RED_COLS = ["red_1", "red_2", "red_3", "red_4", "red_5", "red_6"]
BLUE_COL = "blue"
DATE_COL = "draw_date"
SD_DIGIT_COLS = ["d1", "d2", "d3"]
SD_SUM_COL = "sum"
DLT_FRONT_COLS = ["front_1", "front_2", "front_3", "front_4", "front_5"]
DLT_BACK_COLS = ["back_1", "back_2"]
PLOT_AXIS_LABEL_SIZE = 20
PLOT_TICK_LABEL_SIZE = 18
PLOT_LEGEND_SIZE = 14
PLOT_LINE_WIDTH = 3
PLOT_MARKER_SIZE = 6
MARKOV_DEFAULT_WINDOW = 120
MARKOV_DEFAULT_SMOOTH = 1.0
MARKOV_DEFAULT_TOP_N = 10


def sanitize_filename_fragment(text: str) -> str:
    # 清洗文件名片段，避免生成的图片名包含非法字符
    safe_chars = []
    for char in str(text):
        if char.isalnum() or char in {"_", "-"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_") or "lottery"


def set_plot_context(source_name: str) -> None:
    # 设置当前绘图上下文，用于统一图片命名
    source_stem = sanitize_filename_fragment(Path(source_name).stem)
    st.session_state["plot_source_name"] = source_stem
    st.session_state["plot_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")


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


@st.cache_data(show_spinner=False)
def load_sd_data(file_path: str) -> pd.DataFrame:
    # 读取福彩3D Excel 并基础清洗
    df = pd.read_excel(file_path, engine="openpyxl")
    df["issue"] = normalize_numeric_str(df["issue"], 7)
    for col in SD_DIGIT_COLS:
        df[col] = normalize_numeric_str(df[col], 1)
    df[SD_SUM_COL] = pd.to_numeric(df[SD_SUM_COL], errors="coerce")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_dlt_data(file_path: str) -> pd.DataFrame:
    # 读取大乐透 Excel 并做基础清洗
    df = pd.read_excel(file_path, engine="openpyxl")
    df["issue"] = normalize_numeric_str(df["issue"], 5)
    for col in DLT_FRONT_COLS + DLT_BACK_COLS:
        df[col] = normalize_numeric_str(df[col], 2)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df


def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    # 生成用于计算的数值 DataFrame
    df_num = df.copy()
    for col in RED_COLS + [BLUE_COL]:
        df_num[col] = pd.to_numeric(df_num[col], errors="coerce")
    return df_num


def to_numeric_sd_df(df: pd.DataFrame) -> pd.DataFrame:
    # 福彩3D 数值化 DataFrame
    df_num = df.copy()
    for col in SD_DIGIT_COLS + [SD_SUM_COL]:
        df_num[col] = pd.to_numeric(df_num[col], errors="coerce")
    return df_num


def to_numeric_dlt_df(df: pd.DataFrame) -> pd.DataFrame:
    # 大乐透数值化 DataFrame
    df_num = df.copy()
    for col in DLT_FRONT_COLS + DLT_BACK_COLS:
        df_num[col] = pd.to_numeric(df_num[col], errors="coerce")
    return df_num


def setup_plot_style() -> None:
    # 绘图统一字体与符号设置
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.labelsize"] = PLOT_AXIS_LABEL_SIZE
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titlesize"] = 22
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.labelsize"] = PLOT_TICK_LABEL_SIZE
    plt.rcParams["ytick.labelsize"] = PLOT_TICK_LABEL_SIZE
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = False
    plt.rcParams["ytick.right"] = False
    plt.rcParams["legend.fontsize"] = PLOT_LEGEND_SIZE
    plt.rcParams["lines.linewidth"] = PLOT_LINE_WIDTH
    plt.rcParams["lines.markersize"] = PLOT_MARKER_SIZE


def save_and_show(fig: plt.Figure, name: str) -> Path:
    # 保存图像并返回路径
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    source_name = st.session_state.get("plot_source_name", "lottery")
    timestamp = st.session_state.get("plot_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    safe_name = sanitize_filename_fragment(name)
    output_path = PLOT_DIR / f"{source_name}_{safe_name}_{timestamp}.jpg"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    return output_path


def render_ball_row(reds: Iterable[str], blue: str) -> None:
    # 号码展示的彩色胶囊样式
    red_html = " ".join([f"<span class='ball red'>{n}</span>" for n in reds])
    blue_html = f"<span class='ball blue'>{blue}</span>"
    st.markdown(f"<div class='ball-row'>{red_html} {blue_html}</div>", unsafe_allow_html=True)


def render_dlt_row(fronts: Iterable[str], backs: Iterable[str]) -> None:
    # 大乐透号码展示
    front_html = " ".join([f"<span class='ball red'>{n}</span>" for n in fronts])
    back_html = " ".join([f"<span class='ball blue'>{n}</span>" for n in backs])
    st.markdown(f"<div class='ball-row'>{front_html} {back_html}</div>", unsafe_allow_html=True)


def render_sd_row(digits: Iterable[str], sum_value: int) -> None:
    # 福彩3D 号码展示
    digit_html = " ".join([f"<span class='ball orange'>{n}</span>" for n in digits])
    st.markdown(
        f"<div class='ball-row'>{digit_html}<span class='ball blue'>Sum {sum_value}</span></div>",
        unsafe_allow_html=True,
    )


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


def plot_dlt_front_frequency(df_num: pd.DataFrame, tag: str) -> Path:
    setup_plot_style()
    numbers = pd.to_numeric(df_num[DLT_FRONT_COLS].stack(), errors="coerce").dropna().astype(int)
    counts = numbers.value_counts().reindex(range(1, 36), fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(1, 36), counts.values, color="#e74c3c")
    ax.set_title("Front Area Frequency")
    ax.set_xlabel("Number")
    ax.set_ylabel("Count")
    ax.set_xticks(range(1, 36, 1))
    ax.grid(axis="y", alpha=0.3)
    return save_and_show(fig, f"dlt_front_frequency_{tag}")


def plot_dlt_back_frequency(df_num: pd.DataFrame, tag: str) -> Path:
    setup_plot_style()
    numbers = pd.to_numeric(df_num[DLT_BACK_COLS].stack(), errors="coerce").dropna().astype(int)
    counts = numbers.value_counts().reindex(range(1, 13), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, 13), counts.values, color="#3498db")
    ax.set_title("Back Area Frequency")
    ax.set_xlabel("Number")
    ax.set_ylabel("Count")
    ax.set_xticks(range(1, 13, 1))
    ax.grid(axis="y", alpha=0.3)
    return save_and_show(fig, f"dlt_back_frequency_{tag}")


def plot_dlt_front_trend(df_num: pd.DataFrame, tag: str) -> Path:
    setup_plot_style()
    df_rev = df_num.iloc[::-1].reset_index(drop=True)
    x = np.repeat(np.arange(len(df_rev)), len(DLT_FRONT_COLS))
    y = df_rev[DLT_FRONT_COLS].astype(int).to_numpy().flatten()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(x, y, s=10, alpha=0.6, color="#2c3e50")
    ax.set_title("Front Area Trend")
    ax.set_xlabel("Draw Index (Oldest to Newest)")
    ax.set_ylabel("Number")
    ax.set_yticks(range(1, 36, 2))
    ax.grid(axis="y", alpha=0.2)
    return save_and_show(fig, f"dlt_front_trend_{tag}")


def plot_dlt_back_trend(df_num: pd.DataFrame, tag: str) -> Path:
    setup_plot_style()
    df_rev = df_num.iloc[::-1].reset_index(drop=True)
    x = np.repeat(np.arange(len(df_rev)), len(DLT_BACK_COLS))
    y = df_rev[DLT_BACK_COLS].astype(int).to_numpy().flatten()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(x, y, s=14, alpha=0.7, color="#1f77b4")
    ax.set_title("Back Area Trend")
    ax.set_xlabel("Draw Index (Oldest to Newest)")
    ax.set_ylabel("Number")
    ax.set_yticks(range(1, 13, 1))
    ax.grid(axis="y", alpha=0.2)
    return save_and_show(fig, f"dlt_back_trend_{tag}")


def plot_dlt_sum_trend(df_num: pd.DataFrame, tag: str) -> Path:
    setup_plot_style()
    df_rev = df_num.iloc[::-1].reset_index(drop=True)
    x = np.arange(len(df_rev))
    front_sum = df_rev[DLT_FRONT_COLS].astype(int).sum(axis=1)
    total_sum = front_sum + df_rev[DLT_BACK_COLS].astype(int).sum(axis=1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, front_sum, color="#e67e22", label="Front Sum")
    ax.plot(x, total_sum, color="#16a085", label="Total Sum")
    ax.set_title("Sum Trend")
    ax.set_xlabel("Draw Index (Oldest to Newest)")
    ax.set_ylabel("Sum")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    return save_and_show(fig, f"dlt_sum_trend_{tag}")


def plot_dlt_span_trend(df_num: pd.DataFrame, tag: str) -> Path:
    setup_plot_style()
    df_rev = df_num.iloc[::-1].reset_index(drop=True)
    x = np.arange(len(df_rev))
    span = df_rev[DLT_FRONT_COLS].astype(int).max(axis=1) - df_rev[DLT_FRONT_COLS].astype(int).min(axis=1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, span, color="#8e44ad", linewidth=1.5)
    ax.set_title("Front Area Span Trend")
    ax.set_xlabel("Draw Index (Oldest to Newest)")
    ax.set_ylabel("Span")
    ax.grid(axis="y", alpha=0.2)
    return save_and_show(fig, f"dlt_span_trend_{tag}")


def plot_dlt_odd_even_trend(df_num: pd.DataFrame, tag: str) -> Path:
    setup_plot_style()
    df_rev = df_num.iloc[::-1].reset_index(drop=True)
    x = np.arange(len(df_rev))
    front_vals = df_rev[DLT_FRONT_COLS].astype(int)
    odd_count = (front_vals % 2 == 1).sum(axis=1)
    even_count = 5 - odd_count
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, odd_count, color="#c0392b", label="Odd Count")
    ax.plot(x, even_count, color="#2980b9", label="Even Count")
    ax.set_title("Odd vs Even Trend")
    ax.set_xlabel("Draw Index (Oldest to Newest)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    return save_and_show(fig, f"dlt_odd_even_trend_{tag}")


def plot_sd_position_frequency(df_num: pd.DataFrame, tag: str) -> Path:
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    position_titles = ["Hundreds", "Tens", "Ones"]
    for idx, col in enumerate(SD_DIGIT_COLS):
        counts = df_num[col].astype(int).value_counts().reindex(range(0, 10), fill_value=0)
        axes[idx].bar(range(0, 10), counts.values, color="#f39c12")
        axes[idx].set_title(f"{position_titles[idx]} Position Frequency")
        axes[idx].set_xlabel("Digit")
        axes[idx].set_xticks(range(0, 10))
        axes[idx].grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Count")
    return save_and_show(fig, f"sd_position_frequency_{tag}")


def plot_sd_overall_frequency(df_num: pd.DataFrame, tag: str) -> Path:
    setup_plot_style()
    values = pd.to_numeric(df_num[SD_DIGIT_COLS].stack(), errors="coerce").dropna().astype(int)
    counts = values.value_counts().reindex(range(0, 10), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(0, 10), counts.values, color="#2ecc71")
    ax.set_title("Overall Digit Frequency")
    ax.set_xlabel("Digit")
    ax.set_ylabel("Count")
    ax.set_xticks(range(0, 10))
    ax.grid(axis="y", alpha=0.3)
    return save_and_show(fig, f"sd_overall_frequency_{tag}")


def plot_sd_sum_trend(df_num: pd.DataFrame, tag: str) -> Path:
    setup_plot_style()
    df_rev = df_num.iloc[::-1].reset_index(drop=True)
    x = np.arange(len(df_rev))
    sums = df_rev[SD_SUM_COL].astype(int)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, sums, color="#8e44ad", linewidth=1.5)
    ax.set_title("Sum Trend")
    ax.set_xlabel("Draw Index (Oldest to Newest)")
    ax.set_ylabel("Sum")
    ax.grid(axis="y", alpha=0.2)
    return save_and_show(fig, f"sd_sum_trend_{tag}")


def plot_sd_sum_distribution(df_num: pd.DataFrame, tag: str) -> Path:
    setup_plot_style()
    sums = df_num[SD_SUM_COL].astype(int)
    counts = sums.value_counts().reindex(range(0, 28), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(0, 28), counts.values, color="#9b59b6")
    ax.set_title("Sum Distribution")
    ax.set_xlabel("Sum")
    ax.set_ylabel("Count")
    ax.set_xticks(range(0, 28, 2))
    ax.grid(axis="y", alpha=0.3)
    return save_and_show(fig, f"sd_sum_distribution_{tag}")


def plot_sd_odd_even_trend(df_num: pd.DataFrame, tag: str) -> Path:
    setup_plot_style()
    df_rev = df_num.iloc[::-1].reset_index(drop=True)
    x = np.arange(len(df_rev))
    digits = df_rev[SD_DIGIT_COLS].astype(int)
    odd_count = (digits % 2 == 1).sum(axis=1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, odd_count, color="#c0392b", label="Odd Count")
    ax.plot(x, 3 - odd_count, color="#2980b9", label="Even Count")
    ax.set_title("Odd vs Even Trend")
    ax.set_xlabel("Draw Index (Oldest to Newest)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    return save_and_show(fig, f"sd_odd_even_trend_{tag}")


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
    method_weights: List[float] | None = None,
) -> Tuple[List[int], int]:
    # 综合投票 + 热度微调 + 分区约束
    all_reds = list(range(1, 34))
    red_votes = {n: 0.0 for n in all_reds}
    for idx, item in enumerate(method_results):
        weight = method_weights[idx] if method_weights and idx < len(method_weights) else 1.0
        reds = item["reds"]
        for n in reds:
            red_votes[n] += weight

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

    blue_votes = {n: 0.0 for n in range(1, 17)}
    for idx, item in enumerate(method_results):
        weight = method_weights[idx] if method_weights and idx < len(method_weights) else 1.0
        blue = int(item["blue"])
        blue_votes[blue] += weight

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


def dlt_bucket_index(num: int) -> int:
    # 大乐透前区号码映射到三个区间
    if num <= 12:
        return 0
    if num <= 24:
        return 1
    return 2


def dlt_count_buckets(nums: Iterable[int]) -> List[int]:
    # 大乐透前区区间分布统计
    counts = [0, 0, 0]
    for n in nums:
        counts[dlt_bucket_index(int(n))] += 1
    return counts


def compute_dlt_bucket_target(df_num: pd.DataFrame) -> List[int]:
    # 计算大乐透前区目标区间分布并校正为 5 个号码
    front_vals = df_num[DLT_FRONT_COLS].astype(int)
    bucket_counts: List[List[int]] = []
    for row in front_vals.itertuples(index=False):
        bucket_counts.append(dlt_count_buckets(row))
    if not bucket_counts:
        return [2, 2, 1]
    mean_counts = np.mean(bucket_counts, axis=0)
    target = [int(round(val)) for val in mean_counts]
    while sum(target) > 5:
        idx = target.index(max(target))
        target[idx] -= 1
    while sum(target) < 5:
        idx = target.index(min(target))
        target[idx] += 1
    return target


def compute_dlt_front_stats(df_num: pd.DataFrame) -> Dict[str, float | List[int]]:
    # 大乐透前区结构统计
    front_vals = df_num[DLT_FRONT_COLS].astype(int)
    sum_series = front_vals.sum(axis=1)
    span_series = front_vals.max(axis=1) - front_vals.min(axis=1)
    odd_series = (front_vals % 2 == 1).sum(axis=1)
    return {
        "sum_mean": float(sum_series.mean()),
        "sum_std": float(sum_series.std(ddof=0) or 1.0),
        "span_mean": float(span_series.mean()),
        "span_std": float(span_series.std(ddof=0) or 1.0),
        "odd_mean": float(odd_series.mean()),
        "bucket_target": compute_dlt_bucket_target(df_num),
    }


def pick_top_by_bucket_target(
    scores: Dict[int, float], buckets: List[List[int]], target_counts: List[int], rng: random.Random
) -> List[int]:
    # 按区间目标数量选取高分号码
    selected: List[int] = []
    for bucket, count in zip(buckets, target_counts):
        candidates = bucket[:]
        rng.shuffle(candidates)
        candidates.sort(key=lambda n: scores.get(n, 0.0), reverse=True)
        selected.extend(candidates[:count])
    return selected


def pick_top_by_bucket_target_deterministic(
    scores: Dict[int, float], buckets: List[List[int]], target_counts: List[int]
) -> List[int]:
    # 按区间目标数量稳定选取高分号码，避免额外随机扰动
    selected: List[int] = []
    for bucket, count in zip(buckets, target_counts):
        ranked = sorted(bucket, key=lambda n: (-scores.get(n, 0.0), n))
        selected.extend(ranked[:count])
    return sorted(selected)


def sample_dlt_front_numbers(rng: random.Random, weights: Dict[int, float] | None = None) -> List[int]:
    # 根据权重抽取大乐透前区号码
    if weights:
        numbers = list(weights.keys())
        w = [weights[n] for n in numbers]
        return sorted(weighted_sample_without_replacement(numbers, w, 5, rng))
    return sorted(rng.sample(range(1, 36), 5))


def sample_dlt_back_numbers(rng: random.Random, weights: Dict[int, float] | None = None) -> List[int]:
    # 根据权重抽取大乐透后区号码
    numbers = list(range(1, 13))
    if weights:
        w = [weights[n] for n in numbers]
        return sorted(weighted_sample_without_replacement(numbers, w, 2, rng))
    return sorted(rng.sample(numbers, 2))


def score_dlt_candidate_set(
    nums: List[int],
    target_sum: float,
    target_span: float,
    target_odd: float,
    bucket_target: List[int],
    sum_scale: float,
    span_scale: float,
) -> float:
    # 大乐透前区候选组合评分（越小越好）
    nums_sorted = sorted(nums)
    sum_val = sum(nums_sorted)
    span_val = nums_sorted[-1] - nums_sorted[0]
    odd_count = sum(1 for n in nums_sorted if n % 2 == 1)
    bucket_counts = dlt_count_buckets(nums_sorted)
    consecutive = sum(
        1 for i in range(len(nums_sorted) - 1) if nums_sorted[i + 1] - nums_sorted[i] == 1
    )

    sum_score = abs(sum_val - target_sum) / max(1.0, sum_scale)
    span_score = abs(span_val - target_span) / max(1.0, span_scale)
    odd_score = abs(odd_count - target_odd)
    bucket_score = sum(abs(bucket_counts[i] - bucket_target[i]) for i in range(3))
    consecutive_score = max(0, consecutive - 1)

    return sum_score + span_score + 0.6 * odd_score + 0.4 * bucket_score + 0.2 * consecutive_score


def random_search_dlt_front_set(
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
    # 随机搜索最优大乐透前区组合
    best = sample_dlt_front_numbers(rng, weights)
    best_score = score_dlt_candidate_set(
        best, target_sum, target_span, target_odd, bucket_target, sum_scale, span_scale
    )
    for _ in range(iterations):
        candidate = sample_dlt_front_numbers(rng, weights)
        score = score_dlt_candidate_set(
            candidate, target_sum, target_span, target_odd, bucket_target, sum_scale, span_scale
        )
        if score < best_score:
            best = candidate
            best_score = score
    return best


def pick_low_pmi_set_dlt(
    pmi: Dict[int, Dict[int, float]],
    numbers: List[int],
    bucket_target: List[int],
    rng: random.Random,
) -> List[int]:
    # 贪心选择大乐透前区低互信息组合
    mean_pmi = {n: float(np.mean([pmi[n][m] for m in numbers if m != n])) for n in numbers}
    start = min(mean_pmi, key=mean_pmi.get)
    selected = [start]
    bucket_counts = dlt_count_buckets(selected)

    while len(selected) < 5:
        best = None
        best_score = float("inf")
        for n in numbers:
            if n in selected:
                continue
            bucket_idx = dlt_bucket_index(n)
            if bucket_counts[bucket_idx] >= bucket_target[bucket_idx]:
                continue
            score = float(np.mean([pmi[n][s] for s in selected]))
            if score < best_score:
                best_score = score
                best = n
        if best is None:
            break
        selected.append(best)
        bucket_counts[dlt_bucket_index(best)] += 1

    if len(selected) < 5:
        remaining = [n for n in numbers if n not in selected]
        rng.shuffle(remaining)
        remaining.sort(key=lambda n: mean_pmi.get(n, 0.0))
        for n in remaining:
            if len(selected) == 5:
                break
            bucket_idx = dlt_bucket_index(n)
            if bucket_counts[bucket_idx] >= bucket_target[bucket_idx]:
                continue
            selected.append(n)
            bucket_counts[bucket_idx] += 1

    return sorted(selected)


def pick_dlt_back_pair_by_target(target_sum: float, rng: random.Random) -> List[int]:
    # 选择和值接近目标的后区组合
    pairs: List[Tuple[int, int]] = []
    numbers = list(range(1, 13))
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            pairs.append((numbers[i], numbers[j]))
    best_diff = min(abs((a + b) - target_sum) for a, b in pairs)
    candidates = [pair for pair in pairs if abs((pair[0] + pair[1]) - target_sum) == best_diff]
    rng.shuffle(candidates)
    return sorted(candidates[0])


def build_dlt_ensemble(
    method_results: List[Dict[str, object]],
    recency_scores_front: Dict[int, float],
    recency_scores_back: Dict[int, float],
    bucket_target: List[int],
    rng: random.Random,
    method_weights: List[float] | None = None,
) -> Tuple[List[int], List[int]]:
    # 大乐透综合投票 + 热度微调 + 分区约束
    all_front = list(range(1, 36))
    front_votes = {n: 0.0 for n in all_front}
    for idx, item in enumerate(method_results):
        weight = method_weights[idx] if method_weights and idx < len(method_weights) else 1.0
        fronts = item["fronts"]
        for n in fronts:
            front_votes[n] += weight

    max_recency = max(recency_scores_front.values()) if recency_scores_front else 1.0
    front_scores = {
        n: front_votes[n] + 0.1 * (recency_scores_front.get(n, 0.0) / max_recency)
        for n in all_front
    }

    buckets = [list(range(1, 13)), list(range(13, 25)), list(range(25, 36))]
    selected = pick_top_by_bucket_target(front_scores, buckets, bucket_target, rng)
    bucket_counts = dlt_count_buckets(selected)

    ranked = all_front[:]
    rng.shuffle(ranked)
    ranked.sort(key=lambda n: front_scores.get(n, 0.0), reverse=True)
    for n in ranked:
        if len(selected) >= 5:
            break
        if n in selected:
            continue
        bucket_idx = dlt_bucket_index(n)
        if bucket_counts[bucket_idx] >= bucket_target[bucket_idx]:
            continue
        selected.append(n)
        bucket_counts[bucket_idx] += 1

    back_votes = {n: 0.0 for n in range(1, 13)}
    for idx, item in enumerate(method_results):
        weight = method_weights[idx] if method_weights and idx < len(method_weights) else 1.0
        backs = item["backs"]
        for n in backs:
            back_votes[n] += weight

    max_back_recency = max(recency_scores_back.values()) if recency_scores_back else 1.0
    back_scores = {
        n: back_votes[n] + 0.1 * (recency_scores_back.get(n, 0.0) / max_back_recency)
        for n in range(1, 13)
    }
    back_candidates = list(range(1, 13))
    rng.shuffle(back_candidates)
    back_candidates.sort(key=lambda n: back_scores.get(n, 0.0), reverse=True)
    back_pick = sorted(back_candidates[:2])

    return sorted(selected), back_pick


def compute_ssq_method_weights(
    method_results: List[Dict[str, object]],
    actual_reds: Iterable[int],
    actual_blue: int,
    blue_factor: float = 0.6,
    base_weight: float = 1.0,
) -> List[float]:
    # 根据上一期实际号码为各方法分配权重
    actual_red_set = {int(n) for n in actual_reds}
    actual_blue = int(actual_blue)
    weights: List[float] = []
    for item in method_results:
        reds = {int(n) for n in item["reds"]}
        blue = int(item["blue"])
        red_hit = len(reds & actual_red_set)
        blue_hit = 1 if blue == actual_blue else 0
        score = red_hit + blue_factor * blue_hit
        weights.append(base_weight + score)
    return weights


def compute_dlt_method_weights(
    method_results: List[Dict[str, object]],
    actual_fronts: Iterable[int],
    actual_backs: Iterable[int],
    back_factor: float = 0.6,
    base_weight: float = 1.0,
) -> List[float]:
    # 根据上一期实际号码为大乐透方法分配权重
    actual_front_set = {int(n) for n in actual_fronts}
    actual_back_set = {int(n) for n in actual_backs}
    weights: List[float] = []
    for item in method_results:
        fronts = {int(n) for n in item["fronts"]}
        backs = {int(n) for n in item["backs"]}
        front_hit = len(fronts & actual_front_set)
        back_hit = len(backs & actual_back_set)
        score = front_hit + back_factor * back_hit
        weights.append(base_weight + score)
    return weights


def compute_sd_method_weights(
    method_results: List[Dict[str, object]], actual_digits: Iterable[int], base_weight: float = 1.0
) -> List[float]:
    # 根据上一期实际号码为福彩3D方法分配权重
    actual_list = [int(d) for d in actual_digits]
    weights: List[float] = []
    for item in method_results:
        digits = [int(d) for d in item["digits"]]
        hit = sum(1 for idx, val in enumerate(digits) if idx < len(actual_list) and val == actual_list[idx])
        weights.append(base_weight + hit)
    return weights


def build_ssq_method_results(
    df_recent_num: pd.DataFrame,
    latest_row: pd.Series,
    recency_scores_red: Dict[int, float],
    recency_scores_blue: Dict[int, float],
    red_stats: Dict[str, float | List[int]],
    base_seed: int,
) -> List[Dict[str, object]]:
    # 生成双色球方法明细
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
        df_recent_num, latest_row, recency_scores_red, rng_mirror
    )
    method_results.append(
        {
            "name": "方法十五：镜像映射（对称扰动）",
            "desc": "思路：对最近一期号码做中点对称映射，并用冷号轻微扰动补齐。",
            "reds": reds_o,
            "blue": blue_o,
        }
    )

    return method_results


def build_dlt_method_results(
    df_recent_num: pd.DataFrame,
    latest_row: pd.Series,
    recency_scores_front: Dict[int, float],
    recency_scores_back: Dict[int, float],
    dlt_stats: Dict[str, float | List[int]],
    base_seed: int,
) -> List[Dict[str, object]]:
    # 生成大乐透方法明细
    rng_entropy = random.Random(base_seed + 11)
    rng_gap = random.Random(base_seed + 23)
    rng_cluster = random.Random(base_seed + 29)
    rng_hot = random.Random(base_seed + 31)
    rng_cycle = random.Random(base_seed + 37)
    rng_mirror = random.Random(base_seed + 41)
    rng_markov = random.Random(base_seed + 47)
    rng_bayes = random.Random(base_seed + 53)
    rng_poisson = random.Random(base_seed + 59)
    rng_time = random.Random(base_seed + 61)
    rng_mi = random.Random(base_seed + 67)
    rng_combo = random.Random(base_seed + 71)
    rng_mc = random.Random(base_seed + 73)
    rng_vol = random.Random(base_seed + 79)
    rng_phase = random.Random(base_seed + 83)

    method_results: List[Dict[str, object]] = []

    fronts_a, backs_a = predict_dlt_entropy(df_recent_num, dlt_stats["bucket_target"], rng_entropy)
    method_results.append(
        {
            "name": "方法一：分段熵平衡（冷热反向权重）",
            "desc": "思路：对低频号码赋予更高权重，并按区间抽样。",
            "fronts": fronts_a,
            "backs": backs_a,
        }
    )

    fronts_b, backs_b = predict_dlt_gap_wave(df_recent_num, rng_gap)
    method_results.append(
        {
            "name": "方法二：间隔波动（偏好中等空档）",
            "desc": "思路：偏向出现间隔接近中位数的号码。",
            "fronts": fronts_b,
            "backs": backs_b,
        }
    )

    fronts_c, backs_c = predict_dlt_anti_cluster(df_recent_num, rng_cluster)
    method_results.append(
        {
            "name": "方法三：反聚类协同网络（弱相关组合）",
            "desc": "思路：选择共现次数较少的号码组合。",
            "fronts": fronts_c,
            "backs": backs_c,
        }
    )

    fronts_d, backs_d = predict_dlt_recency_hot(
        df_recent_num, recency_scores_front, recency_scores_back, dlt_stats["bucket_target"], rng_hot
    )
    method_results.append(
        {
            "name": "方法四：指数记忆热度（近期高权重）",
            "desc": "思路：强调近期出现频次较高的号码。",
            "fronts": fronts_d,
            "backs": backs_d,
        }
    )

    fronts_e, backs_e = predict_dlt_cycle_reversion(
        df_recent_num, dlt_stats["bucket_target"], rng_cycle
    )
    method_results.append(
        {
            "name": "方法五：周期回归（间隔接近均值）",
            "desc": "思路：选择间隔接近历史均值的号码。",
            "fronts": fronts_e,
            "backs": backs_e,
        }
    )

    fronts_f, backs_f = predict_dlt_mirror(
        df_recent_num, latest_row, recency_scores_front, recency_scores_back, rng_mirror
    )
    method_results.append(
        {
            "name": "方法六：镜像映射（对称扰动）",
            "desc": "思路：围绕中点对称映射并做轻微扰动。",
            "fronts": fronts_f,
            "backs": backs_f,
        }
    )

    fronts_g, backs_g = predict_dlt_markov(
        df_recent_num, dlt_stats["bucket_target"], rng_markov
    )
    method_results.append(
        {
            "name": "方法七：马尔可夫转移（状态概率）",
            "desc": "思路：依据上一期状态估计下一期转移概率。",
            "fronts": fronts_g,
            "backs": backs_g,
        }
    )

    fronts_h, backs_h = predict_dlt_bayesian(
        df_recent_num, dlt_stats["bucket_target"], rng_bayes
    )
    method_results.append(
        {
            "name": "方法八：贝叶斯更新（后验均值）",
            "desc": "思路：用先验与频次更新后验概率。",
            "fronts": fronts_h,
            "backs": backs_h,
        }
    )

    fronts_i, backs_i = predict_dlt_multinomial_poisson(
        df_recent_num, dlt_stats["bucket_target"], rng_poisson
    )
    method_results.append(
        {
            "name": "方法九：多项/泊松稳定度",
            "desc": "思路：衡量频次与期望次数的贴合度。",
            "fronts": fronts_i,
            "backs": backs_i,
        }
    )

    fronts_j, backs_j = predict_dlt_time_series(df_recent_num, dlt_stats, rng_time)
    method_results.append(
        {
            "name": "方法十：时间序列趋势（和值预测）",
            "desc": "思路：以和值趋势为目标进行组合搜索。",
            "fronts": fronts_j,
            "backs": backs_j,
        }
    )

    fronts_k, backs_k = predict_dlt_mutual_info(df_recent_num, dlt_stats, rng_mi)
    method_results.append(
        {
            "name": "方法十一：互信息网络（弱关联）",
            "desc": "思路：尽量选取互信息较低的号码组合。",
            "fronts": fronts_k,
            "backs": backs_k,
        }
    )

    fronts_l, backs_l = predict_dlt_combo_opt(df_recent_num, dlt_stats, rng_combo)
    method_results.append(
        {
            "name": "方法十二：组合优化（多目标约束）",
            "desc": "思路：同时约束和值、奇偶与区间结构。",
            "fronts": fronts_l,
            "backs": backs_l,
        }
    )

    fronts_m, backs_m = predict_dlt_monte_carlo(df_recent_num, dlt_stats, rng_mc)
    method_results.append(
        {
            "name": "方法十三：Bootstrap/Monte Carlo 模拟",
            "desc": "思路：概率模拟后选择出现频率较高组合。",
            "fronts": fronts_m,
            "backs": backs_m,
        }
    )

    fronts_n, backs_n = predict_dlt_volatility_reversion(df_recent_num, dlt_stats, rng_vol)
    method_results.append(
        {
            "name": "方法十四：波动回归（和值均值回归）",
            "desc": "思路：依据近期和值偏离进行回归预测。",
            "fronts": fronts_n,
            "backs": backs_n,
        }
    )

    fronts_o, backs_o = predict_dlt_phase_space(df_recent_num, dlt_stats, rng_phase)
    method_results.append(
        {
            "name": "方法十五：复杂系统相空间类比",
            "desc": "思路：寻找相似和值轨迹并预测下一步。",
            "fronts": fronts_o,
            "backs": backs_o,
        }
    )

    return method_results


def build_sd_method_results(
    df_recent_num: pd.DataFrame,
    sd_stats: Dict[str, float],
    sd_recency_scores: Dict[str, Dict[int, float]],
    base_seed: int,
) -> List[Dict[str, object]]:
    # 生成福彩3D方法明细
    rng_entropy = random.Random(base_seed + 11)
    rng_gap = random.Random(base_seed + 17)
    rng_markov = random.Random(base_seed + 23)
    rng_bayes = random.Random(base_seed + 29)
    rng_poisson = random.Random(base_seed + 31)
    rng_time = random.Random(base_seed + 37)
    rng_mi = random.Random(base_seed + 41)
    rng_combo = random.Random(base_seed + 43)
    rng_mc = random.Random(base_seed + 47)
    rng_vol = random.Random(base_seed + 53)
    rng_phase = random.Random(base_seed + 59)
    rng_hot = random.Random(base_seed + 61)
    rng_cycle = random.Random(base_seed + 67)
    rng_mirror = random.Random(base_seed + 71)

    sd_results: List[Dict[str, object]] = []

    digits_a = predict_sd_entropy(df_recent_num, rng_entropy)
    sd_results.append(
        {
            "name": "方法一：熵平衡（冷号权重）",
            "desc": "思路：对低频数字赋予更高权重，进行位置抽样。",
            "digits": digits_a,
        }
    )

    digits_b = predict_sd_gap_wave(df_recent_num, rng_gap)
    sd_results.append(
        {
            "name": "方法二：间隔波动（中等空档）",
            "desc": "思路：偏好出现间隔接近中位数的数字。",
            "digits": digits_b,
        }
    )

    digits_c = predict_sd_markov(df_recent_num, rng_markov)
    sd_results.append(
        {
            "name": "方法三：马尔可夫转移（位置状态）",
            "desc": "思路：利用上一期数字的转移概率选择下一期。",
            "digits": digits_c,
        }
    )

    digits_d = predict_sd_bayesian(df_recent_num, rng_bayes)
    sd_results.append(
        {
            "name": "方法四：贝叶斯更新（后验均值）",
            "desc": "思路：用先验与频次更新得到位置后验。",
            "digits": digits_d,
        }
    )

    digits_e = predict_sd_poisson(df_recent_num, rng_poisson)
    sd_results.append(
        {
            "name": "方法五：多项/泊松稳定度",
            "desc": "思路：衡量频次与期望出现次数的贴合度。",
            "digits": digits_e,
        }
    )

    digits_f = predict_sd_time_series(df_recent_num, sd_stats, rng_time)
    sd_results.append(
        {
            "name": "方法六：时间序列趋势（和值预测）",
            "desc": "思路：以和值趋势为目标进行组合搜索。",
            "digits": digits_f,
        }
    )

    digits_g = predict_sd_mutual_info(df_recent_num, sd_stats, rng_mi)
    sd_results.append(
        {
            "name": "方法七：互信息网络（弱关联）",
            "desc": "思路：尽量选取互信息较低的数字组合。",
            "digits": digits_g,
        }
    )

    digits_h = predict_sd_combo_opt(df_recent_num, sd_stats, rng_combo)
    sd_results.append(
        {
            "name": "方法八：组合优化（多目标约束）",
            "desc": "思路：同时约束和值、奇偶与大小结构。",
            "digits": digits_h,
        }
    )

    digits_i = predict_sd_monte_carlo(df_recent_num, rng_mc)
    sd_results.append(
        {
            "name": "方法九：Bootstrap/Monte Carlo 模拟",
            "desc": "思路：概率模拟后选择出现频率最高组合。",
            "digits": digits_i,
        }
    )

    digits_j = predict_sd_volatility(df_recent_num, sd_stats, rng_vol)
    sd_results.append(
        {
            "name": "方法十：波动回归（和值均值回归）",
            "desc": "思路：依据近期和值偏离进行回归预测。",
            "digits": digits_j,
        }
    )

    digits_k = predict_sd_phase_space(df_recent_num, sd_stats, rng_phase)
    sd_results.append(
        {
            "name": "方法十一：复杂系统相空间类比",
            "desc": "思路：寻找相似和值轨迹并预测下一步。",
            "digits": digits_k,
        }
    )

    digits_l = predict_sd_recency_hot(df_recent_num, sd_recency_scores, rng_hot)
    sd_results.append(
        {
            "name": "方法十二：指数记忆热度（近期高权重）",
            "desc": "思路：强调近期出现频次较高的数字。",
            "digits": digits_l,
        }
    )

    digits_m = predict_sd_cycle_reversion(df_recent_num, rng_cycle)
    sd_results.append(
        {
            "name": "方法十三：周期回归（间隔接近均值）",
            "desc": "思路：选择间隔接近均值的数字。",
            "digits": digits_m,
        }
    )

    digits_n = predict_sd_mirror(df_recent_num)
    sd_results.append(
        {
            "name": "方法十四：镜像映射（对称扰动）",
            "desc": "思路：对最新数字做镜像映射。",
            "digits": digits_n,
        }
    )

    return sd_results


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


def compute_markov_transition_frame(
    df_num: pd.DataFrame,
    numbers: Iterable[int],
    cols: List[str],
    smooth: float = 1.0,
) -> pd.DataFrame:
    # 生成集合型号码的一阶马尔科夫转移明细表
    draws: List[set[int]] = []
    for row in df_num[cols].itertuples(index=False):
        draws.append({int(v) for v in row if not pd.isna(v)})
    if len(draws) < 2:
        return pd.DataFrame(
            columns=[
                "number",
                "latest_state",
                "prev_0_next_0",
                "prev_0_next_1",
                "prev_1_next_0",
                "prev_1_next_1",
                "appear_probability",
            ]
        )

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
    rows: List[Dict[str, float | int | str]] = []
    for n in numbers:
        latest_state = 1 if n in latest_set else 0
        prev_next = counts[n][latest_state]
        total = prev_next[0] + prev_next[1] + 2.0 * smooth
        appear_probability = (prev_next[1] + smooth) / total if total > 0 else 0.0
        rows.append(
            {
                "number": int(n),
                "latest_state": "Hit" if latest_state == 1 else "Miss",
                "prev_0_next_0": counts[n][0][0],
                "prev_0_next_1": counts[n][0][1],
                "prev_1_next_0": counts[n][1][0],
                "prev_1_next_1": counts[n][1][1],
                "appear_probability": float(appear_probability),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["appear_probability", "number"], ascending=[False, True]
    ).reset_index(drop=True)


def compute_digit_markov_transition_frame(series: pd.Series, smooth: float = 1.0) -> pd.DataFrame:
    # 生成单个位置数字的一阶马尔科夫转移明细表
    values = series.dropna().astype(int).iloc[::-1].reset_index(drop=True)
    if len(values) < 2:
        return pd.DataFrame(
            columns=["digit", "last_digit", "transition_count", "appear_probability"]
        )

    trans = {i: {j: 0 for j in range(10)} for i in range(10)}
    for prev, nxt in zip(values[:-1], values[1:]):
        trans[int(prev)][int(nxt)] += 1
    last_digit = int(values.iloc[-1])

    rows: List[Dict[str, float | int]] = []
    total = sum(trans[last_digit].values()) + 10.0 * smooth
    for digit in range(10):
        probability = (trans[last_digit][digit] + smooth) / total if total > 0 else 0.0
        rows.append(
            {
                "digit": digit,
                "last_digit": last_digit,
                "transition_count": trans[last_digit][digit],
                "appear_probability": float(probability),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["appear_probability", "digit"], ascending=[False, True]
    ).reset_index(drop=True)


def build_ssq_markov_analysis(
    df_num: pd.DataFrame,
    analysis_window: int,
    smooth: float,
    top_n: int,
) -> Dict[str, object]:
    # 生成双色球马尔科夫专项分析结果
    window_df = df_num.head(min(int(analysis_window), len(df_num))).copy()
    red_frame = compute_markov_transition_frame(window_df, range(1, 34), RED_COLS, smooth)
    blue_frame = compute_markov_transition_frame(window_df, range(1, 17), [BLUE_COL], smooth)

    red_scores = {
        int(row["number"]): float(row["appear_probability"]) for _, row in red_frame.iterrows()
    }
    blue_scores = {
        int(row["number"]): float(row["appear_probability"]) for _, row in blue_frame.iterrows()
    }
    buckets = [list(range(1, 12)), list(range(12, 23)), list(range(23, 34))]
    recommended_reds = pick_top_by_bucket_target_deterministic(red_scores, buckets, [2, 2, 2])
    recommended_blue = min(
        blue_scores,
        key=lambda n: (-blue_scores.get(n, 0.0), n),
    )

    return {
        "window_df": window_df,
        "red_frame": red_frame,
        "blue_frame": blue_frame,
        "recommended_reds": recommended_reds,
        "recommended_blue": int(recommended_blue),
        "top_n": int(top_n),
    }


def build_dlt_markov_analysis(
    df_num: pd.DataFrame,
    analysis_window: int,
    smooth: float,
    top_n: int,
) -> Dict[str, object]:
    # 生成大乐透马尔科夫专项分析结果
    window_df = df_num.head(min(int(analysis_window), len(df_num))).copy()
    front_frame = compute_markov_transition_frame(
        window_df, range(1, 36), DLT_FRONT_COLS, smooth
    )
    back_frame = compute_markov_transition_frame(window_df, range(1, 13), DLT_BACK_COLS, smooth)

    front_scores = {
        int(row["number"]): float(row["appear_probability"]) for _, row in front_frame.iterrows()
    }
    back_scores = {
        int(row["number"]): float(row["appear_probability"]) for _, row in back_frame.iterrows()
    }
    buckets = [list(range(1, 13)), list(range(13, 25)), list(range(25, 36))]
    recommended_fronts = pick_top_by_bucket_target_deterministic(front_scores, buckets, [2, 2, 1])
    ranked_backs = sorted(back_scores, key=lambda n: (-back_scores.get(n, 0.0), n))
    recommended_backs = sorted(ranked_backs[:2])

    return {
        "window_df": window_df,
        "front_frame": front_frame,
        "back_frame": back_frame,
        "recommended_fronts": recommended_fronts,
        "recommended_backs": recommended_backs,
        "top_n": int(top_n),
    }


def build_sd_markov_analysis(
    df_num: pd.DataFrame,
    analysis_window: int,
    smooth: float,
    top_n: int,
) -> Dict[str, object]:
    # 生成福彩3D马尔科夫专项分析结果
    window_df = df_num.head(min(int(analysis_window), len(df_num))).copy()
    position_frames: Dict[str, pd.DataFrame] = {}
    recommended_digits: List[int] = []

    for col in SD_DIGIT_COLS:
        frame = compute_digit_markov_transition_frame(window_df[col], smooth)
        position_frames[col] = frame
        if frame.empty:
            recommended_digits.append(0)
        else:
            recommended_digits.append(int(frame.iloc[0]["digit"]))

    return {
        "window_df": window_df,
        "position_frames": position_frames,
        "recommended_digits": recommended_digits,
        "top_n": int(top_n),
    }


def plot_markov_probability_bar(
    prob_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color: str,
    name: str,
    top_n: int,
    label_width: int = 2,
) -> Path:
    # 绘制马尔科夫概率排序柱状图
    setup_plot_style()
    plot_df = prob_df.head(top_n).copy()
    labels = [f"{int(val):0{label_width}d}" for val in plot_df[x_col]]
    values = plot_df[y_col].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, values, color=color, edgecolor="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Number")
    ax.set_ylabel("Probability")
    ax.grid(axis="y", alpha=0.25)
    return save_and_show(fig, name)


def format_set_markov_frame(frame: pd.DataFrame, number_width: int, top_n: int) -> pd.DataFrame:
    # 格式化集合型号码马尔科夫结果表
    display_df = frame.head(top_n).copy()
    if display_df.empty:
        return display_df
    display_df["number"] = display_df["number"].astype(int).map(
        lambda val: f"{int(val):0{number_width}d}"
    )
    display_df["appear_probability"] = display_df["appear_probability"].map(lambda val: f"{val:.2%}")
    return display_df.rename(
        columns={
            "number": "Number",
            "latest_state": "Latest State",
            "prev_0_next_0": "0→0",
            "prev_0_next_1": "0→1",
            "prev_1_next_0": "1→0",
            "prev_1_next_1": "1→1",
            "appear_probability": "Next Probability",
        }
    )


def format_digit_markov_frame(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    # 格式化福彩3D位置马尔科夫结果表
    display_df = frame.head(top_n).copy()
    if display_df.empty:
        return display_df
    display_df["digit"] = display_df["digit"].astype(int).map(lambda val: f"{val:d}")
    display_df["last_digit"] = display_df["last_digit"].astype(int).map(lambda val: f"{val:d}")
    display_df["appear_probability"] = display_df["appear_probability"].map(lambda val: f"{val:.2%}")
    return display_df.rename(
        columns={
            "digit": "Digit",
            "last_digit": "Last Digit",
            "transition_count": "Transition Count",
            "appear_probability": "Next Probability",
        }
    )


def render_ssq_markov_tab(df_num: pd.DataFrame) -> None:
    # 渲染双色球马尔科夫专项分析页
    st.markdown("**马尔科夫预测专项分析**")
    max_window = max(2, len(df_num))
    control_cols = st.columns(3)
    analysis_window = control_cols[0].number_input(
        "马尔科夫分析期数",
        min_value=2,
        max_value=max_window,
        value=min(MARKOV_DEFAULT_WINDOW, max_window),
        step=10,
        key="ssq_markov_window",
    )
    smooth = control_cols[1].number_input(
        "平滑系数",
        min_value=0.1,
        max_value=5.0,
        value=float(MARKOV_DEFAULT_SMOOTH),
        step=0.1,
        key="ssq_markov_smooth",
    )
    top_n = control_cols[2].slider(
        "概率展示 Top N",
        min_value=3,
        max_value=16,
        value=min(MARKOV_DEFAULT_TOP_N, 16),
        key="ssq_markov_top_n",
    )

    analysis = build_ssq_markov_analysis(df_num, int(analysis_window), float(smooth), int(top_n))
    recommended_reds = analysis["recommended_reds"]
    recommended_blue = analysis["recommended_blue"]
    red_frame = analysis["red_frame"]
    blue_frame = analysis["blue_frame"]

    if red_frame.empty or blue_frame.empty:
        st.warning("历史期数不足，至少需要 2 期数据才能进行马尔科夫转移分析。")
        return

    st.markdown("**马尔科夫推荐号码**")
    render_ball_row([f"{n:02d}" for n in recommended_reds], f"{recommended_blue:02d}")
    st.code(format_ticket(recommended_reds, recommended_blue))
    st.caption("说明：基于上一期命中/未命中状态的一阶转移概率进行排序与筛选。")

    path_red = plot_markov_probability_bar(
        red_frame,
        "number",
        "appear_probability",
        "SSQ Markov Red Probability Ranking",
        "#c0392b",
        "ssq_markov_red_probability",
        int(top_n),
        label_width=2,
    )
    path_blue = plot_markov_probability_bar(
        blue_frame,
        "number",
        "appear_probability",
        "SSQ Markov Blue Probability Ranking",
        "#2980b9",
        "ssq_markov_blue_probability",
        min(int(top_n), 16),
        label_width=2,
    )
    st.caption(f"图片已保存：{path_red}")
    st.caption(f"图片已保存：{path_blue}")

    table_cols = st.columns(2)
    with table_cols[0]:
        st.markdown("**红球转移概率表**")
        st.dataframe(format_set_markov_frame(red_frame, 2, int(top_n)), use_container_width=True)
    with table_cols[1]:
        st.markdown("**蓝球转移概率表**")
        st.dataframe(
            format_set_markov_frame(blue_frame, 2, min(int(top_n), 16)),
            use_container_width=True,
        )


def render_dlt_markov_tab(df_num: pd.DataFrame) -> None:
    # 渲染大乐透马尔科夫专项分析页
    st.markdown("**马尔科夫预测专项分析**")
    max_window = max(2, len(df_num))
    control_cols = st.columns(3)
    analysis_window = control_cols[0].number_input(
        "马尔科夫分析期数",
        min_value=2,
        max_value=max_window,
        value=min(MARKOV_DEFAULT_WINDOW, max_window),
        step=10,
        key="dlt_markov_window",
    )
    smooth = control_cols[1].number_input(
        "平滑系数",
        min_value=0.1,
        max_value=5.0,
        value=float(MARKOV_DEFAULT_SMOOTH),
        step=0.1,
        key="dlt_markov_smooth",
    )
    top_n = control_cols[2].slider(
        "概率展示 Top N",
        min_value=3,
        max_value=12,
        value=min(MARKOV_DEFAULT_TOP_N, 12),
        key="dlt_markov_top_n",
    )

    analysis = build_dlt_markov_analysis(df_num, int(analysis_window), float(smooth), int(top_n))
    recommended_fronts = analysis["recommended_fronts"]
    recommended_backs = analysis["recommended_backs"]
    front_frame = analysis["front_frame"]
    back_frame = analysis["back_frame"]

    if front_frame.empty or back_frame.empty:
        st.warning("历史期数不足，至少需要 2 期数据才能进行马尔科夫转移分析。")
        return

    st.markdown("**马尔科夫推荐号码**")
    render_dlt_row([f"{n:02d}" for n in recommended_fronts], [f"{n:02d}" for n in recommended_backs])
    st.code(format_dlt_ticket(recommended_fronts, recommended_backs))
    st.caption("说明：前区按分区高概率稳定筛选，后区按转移概率排序选取。")

    path_front = plot_markov_probability_bar(
        front_frame,
        "number",
        "appear_probability",
        "DLT Markov Front Probability Ranking",
        "#d35400",
        "dlt_markov_front_probability",
        int(top_n),
        label_width=2,
    )
    path_back = plot_markov_probability_bar(
        back_frame,
        "number",
        "appear_probability",
        "DLT Markov Back Probability Ranking",
        "#2471a3",
        "dlt_markov_back_probability",
        min(int(top_n), 12),
        label_width=2,
    )
    st.caption(f"图片已保存：{path_front}")
    st.caption(f"图片已保存：{path_back}")

    table_cols = st.columns(2)
    with table_cols[0]:
        st.markdown("**前区转移概率表**")
        st.dataframe(format_set_markov_frame(front_frame, 2, int(top_n)), use_container_width=True)
    with table_cols[1]:
        st.markdown("**后区转移概率表**")
        st.dataframe(
            format_set_markov_frame(back_frame, 2, min(int(top_n), 12)),
            use_container_width=True,
        )


def render_sd_markov_tab(df_num: pd.DataFrame) -> None:
    # 渲染福彩3D马尔科夫专项分析页
    st.markdown("**马尔科夫预测专项分析**")
    max_window = max(2, len(df_num))
    control_cols = st.columns(3)
    analysis_window = control_cols[0].number_input(
        "马尔科夫分析期数",
        min_value=2,
        max_value=max_window,
        value=min(MARKOV_DEFAULT_WINDOW, max_window),
        step=10,
        key="sd_markov_window",
    )
    smooth = control_cols[1].number_input(
        "平滑系数",
        min_value=0.1,
        max_value=5.0,
        value=float(MARKOV_DEFAULT_SMOOTH),
        step=0.1,
        key="sd_markov_smooth",
    )
    top_n = control_cols[2].slider(
        "概率展示 Top N",
        min_value=3,
        max_value=10,
        value=min(MARKOV_DEFAULT_TOP_N, 10),
        key="sd_markov_top_n",
    )

    analysis = build_sd_markov_analysis(df_num, int(analysis_window), float(smooth), int(top_n))
    recommended_digits = analysis["recommended_digits"]
    position_frames = analysis["position_frames"]

    if any(frame.empty for frame in position_frames.values()):
        st.warning("历史期数不足，至少需要 2 期数据才能进行马尔科夫转移分析。")
        return

    st.markdown("**马尔科夫推荐号码**")
    render_sd_row([str(d) for d in recommended_digits], sum(recommended_digits))
    st.code(format_sd_ticket(recommended_digits))
    st.caption("说明：分别对百位、十位、个位建立一阶转移概率并独立选择最高概率数字。")

    pos_cols = st.columns(3)
    position_labels = {
        SD_DIGIT_COLS[0]: ("百位", "#e67e22", "sd_markov_pos1_probability"),
        SD_DIGIT_COLS[1]: ("十位", "#27ae60", "sd_markov_pos2_probability"),
        SD_DIGIT_COLS[2]: ("个位", "#8e44ad", "sd_markov_pos3_probability"),
    }
    for idx, col in enumerate(SD_DIGIT_COLS):
        label, color, chart_name = position_labels[col]
        with pos_cols[idx]:
            frame = position_frames[col]
            path_pos = plot_markov_probability_bar(
                frame,
                "digit",
                "appear_probability",
                f"FC3D Markov {label} Probability Ranking",
                color,
                chart_name,
                int(top_n),
                label_width=1,
            )
            st.caption(f"{label}图片已保存：{path_pos}")
            st.dataframe(format_digit_markov_frame(frame, int(top_n)), use_container_width=True)


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


def compute_sd_stats(df_num: pd.DataFrame) -> Dict[str, float]:
    # 福彩3D 结构统计
    digits = df_num[SD_DIGIT_COLS].astype(int)
    sum_series = digits.sum(axis=1)
    odd_series = (digits % 2 == 1).sum(axis=1)
    big_series = (digits >= 5).sum(axis=1)
    return {
        "sum_mean": float(sum_series.mean()),
        "sum_std": float(sum_series.std(ddof=0) or 1.0),
        "odd_mean": float(odd_series.mean()),
        "big_mean": float(big_series.mean()),
    }


def compute_sd_recency_scores(df_num: pd.DataFrame, half_life: float) -> Dict[str, Dict[int, float]]:
    # 福彩3D 位置热度评分
    scores: Dict[str, Dict[int, float]] = {}
    decay = math.log(2) / max(1.0, half_life)
    for col in SD_DIGIT_COLS:
        pos_scores = {d: 0.0 for d in range(10)}
        for idx, val in enumerate(df_num[col].dropna().astype(int)):
            weight = math.exp(-decay * idx)
            if val in pos_scores:
                pos_scores[val] += weight
        scores[col] = pos_scores
    return scores


def pick_digit_from_scores(scores: Dict[int, float], rng: random.Random) -> int:
    # 选择得分最高的数字（含随机打散）
    candidates = list(scores.keys())
    rng.shuffle(candidates)
    candidates.sort(key=lambda n: scores.get(n, 0.0), reverse=True)
    return candidates[0]


def sample_sd_digits(
    rng: random.Random, weights_per_pos: Dict[str, Dict[int, float]] | None = None
) -> List[int]:
    # 按位置权重采样 3D 数字
    digits: List[int] = []
    for col in SD_DIGIT_COLS:
        if weights_per_pos:
            weights = weights_per_pos[col]
            numbers = list(weights.keys())
            w = [weights[n] for n in numbers]
            digits.append(rng.choices(numbers, weights=w, k=1)[0])
        else:
            digits.append(rng.randint(0, 9))
    return digits


def score_sd_candidate(
    digits: List[int],
    target_sum: float,
    target_odd: float,
    target_big: float,
    sum_scale: float,
) -> float:
    # 福彩3D 组合评分（越小越好）
    sum_val = sum(digits)
    odd_count = sum(1 for d in digits if d % 2 == 1)
    big_count = sum(1 for d in digits if d >= 5)
    repeat_penalty = 0.0
    if len(set(digits)) == 1:
        repeat_penalty = 0.8
    elif len(set(digits)) == 2:
        repeat_penalty = 0.3

    sum_score = abs(sum_val - target_sum) / max(1.0, sum_scale)
    odd_score = abs(odd_count - target_odd)
    big_score = abs(big_count - target_big)
    return sum_score + 0.6 * odd_score + 0.6 * big_score + repeat_penalty


def random_search_sd(
    rng: random.Random,
    iterations: int,
    target_sum: float,
    target_odd: float,
    target_big: float,
    sum_scale: float,
    weights_per_pos: Dict[str, Dict[int, float]] | None = None,
) -> List[int]:
    # 随机搜索最优 3D 组合
    best = sample_sd_digits(rng, weights_per_pos)
    best_score = score_sd_candidate(best, target_sum, target_odd, target_big, sum_scale)
    for _ in range(iterations):
        candidate = sample_sd_digits(rng, weights_per_pos)
        score = score_sd_candidate(candidate, target_sum, target_odd, target_big, sum_scale)
        if score < best_score:
            best = candidate
            best_score = score
    return best


def compute_digit_markov_scores(series: pd.Series) -> Dict[int, float]:
    # 计算单位置数字马尔可夫转移
    values = series.dropna().astype(int).iloc[::-1].reset_index(drop=True)
    if len(values) < 2:
        return {d: 0.0 for d in range(10)}

    trans = {i: {j: 1 for j in range(10)} for i in range(10)}
    for prev, nxt in zip(values[:-1], values[1:]):
        trans[int(prev)][int(nxt)] += 1
    last_digit = int(values.iloc[-1])
    return {d: float(trans[last_digit][d]) for d in range(10)}


def compute_digit_pmi_matrix(df_num: pd.DataFrame) -> Dict[int, Dict[int, float]]:
    # 计算数字间互信息矩阵
    total = max(1, len(df_num))
    count = {d: 0 for d in range(10)}
    co_counts: Dict[Tuple[int, int], int] = {}

    for row in df_num[SD_DIGIT_COLS].itertuples(index=False):
        digits = [int(v) for v in row if not pd.isna(v)]
        for d in digits:
            count[d] += 1
        for i in range(len(digits)):
            for j in range(i + 1, len(digits)):
                a, b = digits[i], digits[j]
                if a > b:
                    a, b = b, a
                co_counts[(a, b)] = co_counts.get((a, b), 0) + 1

    pmi = {d: {} for d in range(10)}
    for i in range(10):
        for j in range(10):
            if i == j:
                pmi[i][j] = 0.0
                continue
            a, b = (i, j) if i < j else (j, i)
            p_a = (count[i] + 1) / (total + 2)
            p_b = (count[j] + 1) / (total + 2)
            p_ab = (co_counts.get((a, b), 0) + 1) / (total + 2)
            pmi[i][j] = math.log(p_ab / (p_a * p_b))
    return pmi


def build_sd_ensemble(
    method_results: List[Dict[str, object]],
    recency_scores: Dict[str, Dict[int, float]],
    rng: random.Random,
    method_weights: List[float] | None = None,
) -> List[int]:
    # 福彩3D 综合投票
    votes = {col: {d: 0.0 for d in range(10)} for col in SD_DIGIT_COLS}
    for idx, item in enumerate(method_results):
        weight = method_weights[idx] if method_weights and idx < len(method_weights) else 1.0
        digits = item["digits"]
        for idx, col in enumerate(SD_DIGIT_COLS):
            votes[col][int(digits[idx])] += weight

    result: List[int] = []
    for col in SD_DIGIT_COLS:
        max_recency = max(recency_scores[col].values()) if recency_scores[col] else 1.0
        scores = {
            d: votes[col][d] + 0.1 * (recency_scores[col].get(d, 0.0) / max_recency)
            for d in range(10)
        }
        result.append(pick_digit_from_scores(scores, rng))
    return result


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


def predict_dlt_entropy(
    df_num: pd.DataFrame, bucket_target: List[int], rng: random.Random
) -> Tuple[List[int], List[int]]:
    # 大乐透方法一：分段熵平衡（冷热反向权重）
    all_numbers = list(range(1, 36))
    freq = pd.to_numeric(df_num[DLT_FRONT_COLS].stack(), errors="coerce").dropna().astype(int)
    freq_counts = freq.value_counts().to_dict()
    weights = {n: 1.0 / (freq_counts.get(n, 0) + 1) for n in all_numbers}

    buckets = [list(range(1, 13)), list(range(13, 25)), list(range(25, 36))]
    chosen: List[int] = []
    for bucket, count in zip(buckets, bucket_target):
        bucket_weights = [weights[n] for n in bucket]
        chosen.extend(weighted_sample_without_replacement(bucket, bucket_weights, count, rng))

    back_freq = pd.to_numeric(df_num[DLT_BACK_COLS].stack(), errors="coerce").dropna().astype(int)
    back_counts = back_freq.value_counts().to_dict()
    back_weights = {n: 1.0 / (back_counts.get(n, 0) + 1) for n in range(1, 13)}
    back_pick = sample_dlt_back_numbers(rng, back_weights)
    return sorted(chosen), sorted(back_pick)


def predict_dlt_gap_wave(df_num: pd.DataFrame, rng: random.Random) -> Tuple[List[int], List[int]]:
    # 大乐透方法二：间隔波动（偏好中等空档）
    numbers = list(range(1, 36))
    gaps = compute_gaps(df_num, numbers, DLT_FRONT_COLS)
    gap_values = list(gaps.values())
    median_gap = sorted(gap_values)[len(gap_values) // 2]
    sigma = max(1.0, len(df_num) / 6.0)
    weights = []
    for n in numbers:
        score = math.exp(-((gaps[n] - median_gap) ** 2) / (2 * sigma**2))
        weights.append(score)
    chosen = weighted_sample_without_replacement(numbers, weights, 5, rng)

    back_numbers = list(range(1, 13))
    back_gaps = compute_gaps(df_num, back_numbers, DLT_BACK_COLS)
    back_values = list(back_gaps.values())
    back_median = sorted(back_values)[len(back_values) // 2]
    back_sigma = max(1.0, len(df_num) / 6.0)
    back_weights = [
        math.exp(-((back_gaps[n] - back_median) ** 2) / (2 * back_sigma**2))
        for n in back_numbers
    ]
    back_pick = weighted_sample_without_replacement(back_numbers, back_weights, 2, rng)
    return sorted(chosen), sorted(back_pick)


def predict_dlt_anti_cluster(df_num: pd.DataFrame, rng: random.Random) -> Tuple[List[int], List[int]]:
    # 大乐透方法三：反聚类协同网络（弱相关组合）
    size = 36
    co = [[0 for _ in range(size)] for _ in range(size)]
    for row in df_num[DLT_FRONT_COLS].itertuples(index=False):
        nums = sorted({int(v) for v in row if not pd.isna(v)})
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                co[nums[i]][nums[j]] += 1
                co[nums[j]][nums[i]] += 1

    strength = {n: sum(co[n]) for n in range(1, 36)}
    selected = [max(strength, key=strength.get)]
    while len(selected) < 5:
        candidates = [n for n in range(1, 36) if n not in selected]
        scores = {n: sum(co[n][s] for s in selected) for n in candidates}
        best_score = min(scores.values())
        best_candidates = [n for n, s in scores.items() if s == best_score]
        rng.shuffle(best_candidates)
        best_candidates.sort(key=lambda n: strength[n], reverse=True)
        selected.append(best_candidates[0])

    back_counts = get_number_counts(df_num, range(1, 13), DLT_BACK_COLS)
    back_candidates = list(range(1, 13))
    rng.shuffle(back_candidates)
    back_candidates.sort(key=lambda n: back_counts.get(n, 0), reverse=True)
    return sorted(selected), sorted(back_candidates[:2])


def predict_dlt_recency_hot(
    df_num: pd.DataFrame,
    recency_scores_front: Dict[int, float],
    recency_scores_back: Dict[int, float],
    bucket_target: List[int],
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    # 大乐透方法四：指数记忆热度（近期高权重）
    buckets = [list(range(1, 13)), list(range(13, 25)), list(range(25, 36))]
    chosen = pick_top_by_bucket_target(recency_scores_front, buckets, bucket_target, rng)

    back_candidates = list(range(1, 13))
    rng.shuffle(back_candidates)
    back_candidates.sort(key=lambda n: recency_scores_back.get(n, 0.0), reverse=True)
    return sorted(chosen), sorted(back_candidates[:2])


def predict_dlt_cycle_reversion(
    df_num: pd.DataFrame, bucket_target: List[int], rng: random.Random
) -> Tuple[List[int], List[int]]:
    # 大乐透方法五：周期回归（当前间隔接近历史均值）
    numbers = list(range(1, 36))
    avg_gap, current_gap = compute_gap_stats(df_num, numbers, DLT_FRONT_COLS)
    scores = {n: 1.0 / (abs(current_gap[n] - avg_gap[n]) + 1.0) for n in numbers}

    buckets = [list(range(1, 13)), list(range(13, 25)), list(range(25, 36))]
    chosen = pick_top_by_bucket_target(scores, buckets, bucket_target, rng)

    back_numbers = list(range(1, 13))
    back_avg, back_current = compute_gap_stats(df_num, back_numbers, DLT_BACK_COLS)
    back_scores = {n: 1.0 / (abs(back_current[n] - back_avg[n]) + 1.0) for n in back_numbers}
    back_candidates = back_numbers[:]
    rng.shuffle(back_candidates)
    back_candidates.sort(key=lambda n: back_scores.get(n, 0.0), reverse=True)
    return sorted(chosen), sorted(back_candidates[:2])


def predict_dlt_mirror(
    df_num: pd.DataFrame,
    latest_row: pd.Series,
    recency_scores_front: Dict[int, float],
    recency_scores_back: Dict[int, float],
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    # 大乐透方法六：镜像映射（围绕中点对称并做轻微扰动）
    latest_fronts = [int(latest_row[col]) for col in DLT_FRONT_COLS]
    mirrored = [36 - n for n in latest_fronts]
    chosen: List[int] = []

    for val in mirrored:
        candidate = val
        if candidate in chosen:
            options = [candidate - 1, candidate + 1]
            options = [o for o in options if 1 <= o <= 35 and o not in chosen]
            if options:
                options.sort(key=lambda n: recency_scores_front.get(n, 0.0))
                candidate = options[0]
            else:
                candidate = None
        if candidate is not None and candidate not in chosen:
            chosen.append(candidate)

    if len(chosen) < 5:
        cold_numbers = list(range(1, 36))
        rng.shuffle(cold_numbers)
        cold_numbers.sort(key=lambda n: recency_scores_front.get(n, 0.0))
        for n in cold_numbers:
            if n not in chosen:
                chosen.append(n)
            if len(chosen) == 5:
                break

    latest_backs = [int(latest_row[col]) for col in DLT_BACK_COLS]
    back_mirrored = [13 - n for n in latest_backs]
    back_chosen: List[int] = []
    for val in back_mirrored:
        candidate = val
        if candidate in back_chosen:
            options = [candidate - 1, candidate + 1]
            options = [o for o in options if 1 <= o <= 12 and o not in back_chosen]
            if options:
                options.sort(key=lambda n: recency_scores_back.get(n, 0.0))
                candidate = options[0]
            else:
                candidate = None
        if candidate is not None and candidate not in back_chosen:
            back_chosen.append(candidate)

    if len(back_chosen) < 2:
        cold_back = list(range(1, 13))
        rng.shuffle(cold_back)
        cold_back.sort(key=lambda n: recency_scores_back.get(n, 0.0))
        for n in cold_back:
            if n not in back_chosen:
                back_chosen.append(n)
            if len(back_chosen) == 2:
                break

    return sorted(chosen), sorted(back_chosen)


def predict_dlt_markov(
    df_num: pd.DataFrame, bucket_target: List[int], rng: random.Random
) -> Tuple[List[int], List[int]]:
    # 大乐透方法七：马尔可夫转移
    front_scores = compute_markov_scores(df_num, range(1, 36), DLT_FRONT_COLS)
    buckets = [list(range(1, 13)), list(range(13, 25)), list(range(25, 36))]
    fronts = pick_top_by_bucket_target(front_scores, buckets, bucket_target, rng)

    back_scores = compute_markov_scores(df_num, range(1, 13), DLT_BACK_COLS)
    back_candidates = list(range(1, 13))
    rng.shuffle(back_candidates)
    back_candidates.sort(key=lambda n: back_scores.get(n, 0.0), reverse=True)
    return sorted(fronts), sorted(back_candidates[:2])


def predict_dlt_bayesian(
    df_num: pd.DataFrame, bucket_target: List[int], rng: random.Random
) -> Tuple[List[int], List[int]]:
    # 大乐透方法八：贝叶斯更新
    draws = len(df_num)
    alpha, beta = 1.0, 1.0
    front_counts = get_number_counts(df_num, range(1, 36), DLT_FRONT_COLS)
    front_scores = {
        n: (alpha + front_counts[n]) / (alpha + beta + draws) for n in range(1, 36)
    }
    buckets = [list(range(1, 13)), list(range(13, 25)), list(range(25, 36))]
    fronts = pick_top_by_bucket_target(front_scores, buckets, bucket_target, rng)

    back_counts = get_number_counts(df_num, range(1, 13), DLT_BACK_COLS)
    back_scores = {
        n: (alpha + back_counts[n]) / (alpha + beta + draws) for n in range(1, 13)
    }
    back_candidates = list(range(1, 13))
    rng.shuffle(back_candidates)
    back_candidates.sort(key=lambda n: back_scores.get(n, 0.0), reverse=True)
    return sorted(fronts), sorted(back_candidates[:2])


def predict_dlt_multinomial_poisson(
    df_num: pd.DataFrame, bucket_target: List[int], rng: random.Random
) -> Tuple[List[int], List[int]]:
    # 大乐透方法九：多项分布/泊松稳定度
    draws = len(df_num)
    front_counts = get_number_counts(df_num, range(1, 36), DLT_FRONT_COLS)
    expected_front = draws * 5 / 35
    front_scores = {n: poisson_score(front_counts[n], expected_front) for n in range(1, 36)}
    buckets = [list(range(1, 13)), list(range(13, 25)), list(range(25, 36))]
    fronts = pick_top_by_bucket_target(front_scores, buckets, bucket_target, rng)

    back_counts = get_number_counts(df_num, range(1, 13), DLT_BACK_COLS)
    expected_back = draws * 2 / 12
    back_scores = {n: poisson_score(back_counts[n], expected_back) for n in range(1, 13)}
    back_candidates = list(range(1, 13))
    rng.shuffle(back_candidates)
    back_candidates.sort(key=lambda n: back_scores.get(n, 0.0), reverse=True)
    return sorted(fronts), sorted(back_candidates[:2])


def predict_dlt_time_series(
    df_num: pd.DataFrame, dlt_stats: Dict[str, float | List[int]], rng: random.Random
) -> Tuple[List[int], List[int]]:
    # 大乐透方法十：时间序列趋势（和值预测）
    front_vals = df_num[DLT_FRONT_COLS].astype(int)
    sum_series = front_vals.sum(axis=1).iloc[::-1].reset_index(drop=True)
    window = min(30, len(sum_series))
    trend = sum_series.rolling(window=window, min_periods=max(5, window // 2)).mean()
    target_sum = float(trend.iloc[-1]) if not trend.empty else float(dlt_stats["sum_mean"])

    fronts = random_search_dlt_front_set(
        rng=rng,
        iterations=2000,
        target_sum=target_sum,
        target_span=float(dlt_stats["span_mean"]),
        target_odd=float(dlt_stats["odd_mean"]),
        bucket_target=dlt_stats["bucket_target"],
        sum_scale=float(dlt_stats["sum_std"]),
        span_scale=float(dlt_stats["span_std"]),
    )

    back_sum_series = df_num[DLT_BACK_COLS].astype(int).sum(axis=1).iloc[::-1].reset_index(drop=True)
    back_window = min(20, len(back_sum_series))
    back_trend = back_sum_series.rolling(window=back_window, min_periods=max(3, back_window // 2)).mean()
    back_target = float(back_trend.iloc[-1]) if not back_trend.empty else float(back_sum_series.mean())
    back_pick = pick_dlt_back_pair_by_target(back_target, rng)
    return sorted(fronts), sorted(back_pick)


def predict_dlt_mutual_info(
    df_num: pd.DataFrame, dlt_stats: Dict[str, float | List[int]], rng: random.Random
) -> Tuple[List[int], List[int]]:
    # 大乐透方法十一：互信息网络（弱相关组合）
    numbers = list(range(1, 36))
    pmi = compute_pmi_matrix(df_num, numbers, DLT_FRONT_COLS)
    fronts = pick_low_pmi_set_dlt(pmi, numbers, dlt_stats["bucket_target"], rng)

    back_counts = get_number_counts(df_num, range(1, 13), DLT_BACK_COLS)
    back_candidates = list(range(1, 13))
    rng.shuffle(back_candidates)
    back_candidates.sort(key=lambda n: back_counts.get(n, 0), reverse=True)
    return sorted(fronts), sorted(back_candidates[:2])


def predict_dlt_combo_opt(
    df_num: pd.DataFrame, dlt_stats: Dict[str, float | List[int]], rng: random.Random
) -> Tuple[List[int], List[int]]:
    # 大乐透方法十二：组合优化（多目标约束）
    fronts = random_search_dlt_front_set(
        rng=rng,
        iterations=3000,
        target_sum=float(dlt_stats["sum_mean"]),
        target_span=float(dlt_stats["span_mean"]),
        target_odd=float(dlt_stats["odd_mean"]),
        bucket_target=dlt_stats["bucket_target"],
        sum_scale=float(dlt_stats["sum_std"]),
        span_scale=float(dlt_stats["span_std"]),
    )
    back_sum_mean = float(df_num[DLT_BACK_COLS].astype(int).sum(axis=1).mean())
    back_pick = pick_dlt_back_pair_by_target(back_sum_mean, rng)
    return sorted(fronts), sorted(back_pick)


def predict_dlt_monte_carlo(
    df_num: pd.DataFrame, dlt_stats: Dict[str, float | List[int]], rng: random.Random
) -> Tuple[List[int], List[int]]:
    # 大乐透方法十三：Bootstrap/Monte Carlo 模拟
    numbers = list(range(1, 36))
    back_numbers = list(range(1, 13))
    front_counts = get_number_counts(df_num, numbers, DLT_FRONT_COLS)
    back_counts = get_number_counts(df_num, back_numbers, DLT_BACK_COLS)

    front_weights = [front_counts[n] + 1 for n in numbers]
    back_weights = [back_counts[n] + 1 for n in back_numbers]
    sim_front = {n: 0 for n in numbers}
    sim_back = {n: 0 for n in back_numbers}

    for _ in range(2000):
        fronts = weighted_sample_without_replacement(numbers, front_weights, 5, rng)
        for n in fronts:
            sim_front[n] += 1
        backs = weighted_sample_without_replacement(back_numbers, back_weights, 2, rng)
        for n in backs:
            sim_back[n] += 1

    buckets = [list(range(1, 13)), list(range(13, 25)), list(range(25, 36))]
    fronts = pick_top_by_bucket_target(sim_front, buckets, dlt_stats["bucket_target"], rng)
    back_candidates = list(range(1, 13))
    rng.shuffle(back_candidates)
    back_candidates.sort(key=lambda n: sim_back.get(n, 0), reverse=True)
    return sorted(fronts), sorted(back_candidates[:2])


def predict_dlt_volatility_reversion(
    df_num: pd.DataFrame, dlt_stats: Dict[str, float | List[int]], rng: random.Random
) -> Tuple[List[int], List[int]]:
    # 大乐透方法十四：波动回归（和值均值回归）
    front_vals = df_num[DLT_FRONT_COLS].astype(int)
    sum_series = front_vals.sum(axis=1).iloc[::-1].reset_index(drop=True)
    mean_val = float(sum_series.mean())
    std_val = float(sum_series.std(ddof=0) or 1.0)
    last_sum = float(sum_series.iloc[-1]) if not sum_series.empty else mean_val
    shift = 0.3 * std_val
    target_sum = mean_val - shift if last_sum > mean_val else mean_val + shift

    fronts = random_search_dlt_front_set(
        rng=rng,
        iterations=2000,
        target_sum=target_sum,
        target_span=float(dlt_stats["span_mean"]),
        target_odd=float(dlt_stats["odd_mean"]),
        bucket_target=dlt_stats["bucket_target"],
        sum_scale=float(dlt_stats["sum_std"]),
        span_scale=float(dlt_stats["span_std"]),
    )
    back_sum_mean = float(df_num[DLT_BACK_COLS].astype(int).sum(axis=1).mean())
    back_pick = pick_dlt_back_pair_by_target(back_sum_mean, rng)
    return sorted(fronts), sorted(back_pick)


def predict_dlt_phase_space(
    df_num: pd.DataFrame, dlt_stats: Dict[str, float | List[int]], rng: random.Random
) -> Tuple[List[int], List[int]]:
    # 大乐透方法十五：复杂系统相空间类比
    front_vals = df_num[DLT_FRONT_COLS].astype(int)
    sum_series = front_vals.sum(axis=1).iloc[::-1].reset_index(drop=True)
    embed_dim = 3
    target_sum = float(dlt_stats["sum_mean"])
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

    fronts = random_search_dlt_front_set(
        rng=rng,
        iterations=2000,
        target_sum=target_sum,
        target_span=float(dlt_stats["span_mean"]),
        target_odd=float(dlt_stats["odd_mean"]),
        bucket_target=dlt_stats["bucket_target"],
        sum_scale=float(dlt_stats["sum_std"]),
        span_scale=float(dlt_stats["span_std"]),
    )
    back_sum_series = df_num[DLT_BACK_COLS].astype(int).sum(axis=1).iloc[::-1].reset_index(drop=True)
    back_target = float(back_sum_series.tail(10).mean()) if not back_sum_series.empty else 10.0
    back_pick = pick_dlt_back_pair_by_target(back_target, rng)
    return sorted(fronts), sorted(back_pick)


def predict_sd_entropy(df_num: pd.DataFrame, rng: random.Random) -> List[int]:
    # 3D 方法一：熵平衡（冷号权重）
    weights_per_pos: Dict[str, Dict[int, float]] = {}
    for col in SD_DIGIT_COLS:
        counts = df_num[col].astype(int).value_counts().to_dict()
        weights_per_pos[col] = {d: 1.0 / (counts.get(d, 0) + 1) for d in range(10)}
    return sample_sd_digits(rng, weights_per_pos)


def predict_sd_gap_wave(df_num: pd.DataFrame, rng: random.Random) -> List[int]:
    # 3D 方法二：间隔波动（偏好中等空档）
    weights_per_pos: Dict[str, Dict[int, float]] = {}
    sigma = max(1.0, len(df_num) / 8.0)
    for col in SD_DIGIT_COLS:
        gaps = compute_gaps(df_num, range(10), [col])
        gap_values = list(gaps.values())
        median_gap = sorted(gap_values)[len(gap_values) // 2]
        weights_per_pos[col] = {
            d: math.exp(-((gaps[d] - median_gap) ** 2) / (2 * sigma**2)) for d in range(10)
        }
    return sample_sd_digits(rng, weights_per_pos)


def predict_sd_markov(df_num: pd.DataFrame, rng: random.Random) -> List[int]:
    # 3D 方法三：马尔可夫转移
    digits: List[int] = []
    for col in SD_DIGIT_COLS:
        scores = compute_digit_markov_scores(df_num[col])
        digits.append(pick_digit_from_scores(scores, rng))
    return digits


def predict_sd_bayesian(df_num: pd.DataFrame, rng: random.Random) -> List[int]:
    # 3D 方法四：贝叶斯更新
    digits: List[int] = []
    total = len(df_num)
    alpha = 1.0
    for col in SD_DIGIT_COLS:
        counts = df_num[col].astype(int).value_counts().to_dict()
        scores = {d: (alpha + counts.get(d, 0)) / (alpha * 10 + total) for d in range(10)}
        digits.append(pick_digit_from_scores(scores, rng))
    return digits


def predict_sd_poisson(df_num: pd.DataFrame, rng: random.Random) -> List[int]:
    # 3D 方法五：多项/泊松稳定度
    digits: List[int] = []
    total = len(df_num)
    expected = total / 10.0
    for col in SD_DIGIT_COLS:
        counts = df_num[col].astype(int).value_counts().to_dict()
        scores = {d: poisson_score(counts.get(d, 0), expected) for d in range(10)}
        digits.append(pick_digit_from_scores(scores, rng))
    return digits


def predict_sd_time_series(df_num: pd.DataFrame, stats: Dict[str, float], rng: random.Random) -> List[int]:
    # 3D 方法六：时间序列趋势（和值预测）
    sum_series = df_num[SD_SUM_COL].astype(int).iloc[::-1].reset_index(drop=True)
    window = min(30, len(sum_series))
    trend = sum_series.rolling(window=window, min_periods=max(5, window // 2)).mean()
    target_sum = float(trend.iloc[-1]) if not trend.empty else stats["sum_mean"]
    return random_search_sd(
        rng=rng,
        iterations=2000,
        target_sum=target_sum,
        target_odd=stats["odd_mean"],
        target_big=stats["big_mean"],
        sum_scale=stats["sum_std"],
    )


def predict_sd_mutual_info(df_num: pd.DataFrame, stats: Dict[str, float], rng: random.Random) -> List[int]:
    # 3D 方法七：互信息网络（弱关联）
    pmi = compute_digit_pmi_matrix(df_num)
    target_sum = stats["sum_mean"]
    sum_scale = stats["sum_std"]

    best_digits = [0, 0, 0]
    best_score = float("inf")
    for d1 in range(10):
        for d2 in range(10):
            for d3 in range(10):
                pmi_score = pmi[d1][d2] + pmi[d1][d3] + pmi[d2][d3]
                sum_score = abs((d1 + d2 + d3) - target_sum) / max(1.0, sum_scale)
                score = pmi_score + 0.2 * sum_score + rng.random() * 1e-6
                if score < best_score:
                    best_score = score
                    best_digits = [d1, d2, d3]
    return best_digits


def predict_sd_combo_opt(df_num: pd.DataFrame, stats: Dict[str, float], rng: random.Random) -> List[int]:
    # 3D 方法八：组合优化（多目标约束）
    return random_search_sd(
        rng=rng,
        iterations=3000,
        target_sum=stats["sum_mean"],
        target_odd=stats["odd_mean"],
        target_big=stats["big_mean"],
        sum_scale=stats["sum_std"],
    )


def predict_sd_monte_carlo(df_num: pd.DataFrame, rng: random.Random) -> List[int]:
    # 3D 方法九：Bootstrap/Monte Carlo 模拟
    counts_per_pos = {
        col: df_num[col].astype(int).value_counts().to_dict() for col in SD_DIGIT_COLS
    }
    sim_counts: Dict[Tuple[int, int, int], int] = {}
    for _ in range(3000):
        digits = []
        for col in SD_DIGIT_COLS:
            numbers = list(range(10))
            weights = [counts_per_pos[col].get(n, 0) + 1 for n in numbers]
            digits.append(rng.choices(numbers, weights=weights, k=1)[0])
        key = (digits[0], digits[1], digits[2])
        sim_counts[key] = sim_counts.get(key, 0) + 1
    best = max(sim_counts, key=sim_counts.get)
    return [int(best[0]), int(best[1]), int(best[2])]


def predict_sd_volatility(df_num: pd.DataFrame, stats: Dict[str, float], rng: random.Random) -> List[int]:
    # 3D 方法十：波动回归（和值均值回归）
    sum_series = df_num[SD_SUM_COL].astype(int).iloc[::-1].reset_index(drop=True)
    mean_val = float(sum_series.mean())
    std_val = float(sum_series.std(ddof=0) or 1.0)
    last_sum = float(sum_series.iloc[-1]) if not sum_series.empty else mean_val
    target_sum = mean_val - 0.3 * std_val if last_sum > mean_val else mean_val + 0.3 * std_val
    return random_search_sd(
        rng=rng,
        iterations=2000,
        target_sum=target_sum,
        target_odd=stats["odd_mean"],
        target_big=stats["big_mean"],
        sum_scale=stats["sum_std"],
    )


def predict_sd_phase_space(df_num: pd.DataFrame, stats: Dict[str, float], rng: random.Random) -> List[int]:
    # 3D 方法十一：复杂系统相空间类比
    sum_series = df_num[SD_SUM_COL].astype(int).iloc[::-1].reset_index(drop=True)
    target_sum = stats["sum_mean"]
    embed_dim = 3
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
    return random_search_sd(
        rng=rng,
        iterations=2000,
        target_sum=target_sum,
        target_odd=stats["odd_mean"],
        target_big=stats["big_mean"],
        sum_scale=stats["sum_std"],
    )


def predict_sd_recency_hot(
    df_num: pd.DataFrame, recency_scores: Dict[str, Dict[int, float]], rng: random.Random
) -> List[int]:
    # 3D 方法十二：指数记忆热度（近期高权重）
    digits: List[int] = []
    for col in SD_DIGIT_COLS:
        digits.append(pick_digit_from_scores(recency_scores[col], rng))
    return digits


def predict_sd_cycle_reversion(df_num: pd.DataFrame, rng: random.Random) -> List[int]:
    # 3D 方法十三：周期回归（间隔接近均值）
    digits: List[int] = []
    for col in SD_DIGIT_COLS:
        avg_gap, current_gap = compute_gap_stats(df_num, range(10), [col])
        scores = {d: 1.0 / (abs(current_gap[d] - avg_gap[d]) + 1.0) for d in range(10)}
        digits.append(pick_digit_from_scores(scores, rng))
    return digits


def predict_sd_mirror(df_num: pd.DataFrame) -> List[int]:
    # 3D 方法十四：镜像映射（对称扰动）
    latest = df_num.iloc[0]
    digits = [9 - int(latest[col]) for col in SD_DIGIT_COLS]
    return digits


def format_sd_ticket(digits: List[int]) -> str:
    # 输出 3D 格式号码
    return " ".join(str(d) for d in digits)


def format_dlt_ticket(fronts: List[int], backs: List[int]) -> str:
    # 输出大乐透格式号码
    front_text = " ".join(f"{n:02d}" for n in fronts)
    back_text = " ".join(f"{n:02d}" for n in backs)
    return f"{front_text} + {back_text}"


def format_ticket(reds: List[int], blue: int) -> str:
    # 输出标准号码格式
    red_text = " ".join(f"{n:02d}" for n in reds)
    return f"{red_text} + {blue:02d}"


def get_base_seed_from_row(row: pd.Series) -> int:
    # 生成回测所需的随机种子
    issue_seed = int(str(row["issue"]).lstrip("0") or "0")
    date_seed = int(row[DATE_COL].strftime("%Y%m%d")) if pd.notna(row[DATE_COL]) else 0
    return issue_seed + date_seed


@st.cache_data(show_spinner=False)
def run_ssq_backtest(df: pd.DataFrame, backtest_periods: int, train_window: int) -> pd.DataFrame:
    # 双色球回测统计
    df_sorted = df.sort_values("issue", ascending=False).reset_index(drop=True)
    max_periods = len(df_sorted) - train_window - 1
    periods = min(int(backtest_periods), int(max_periods))
    if periods <= 0:
        return pd.DataFrame()

    stats: Dict[str, Dict[str, List[int]]] = {}
    method_names: List[str] | None = None

    for idx in range(periods):
        train = df_sorted.iloc[idx + 1 : idx + 1 + train_window]
        if train.empty:
            continue
        train_num = to_numeric_df(train)
        if train_num.empty:
            continue

        base_seed = get_base_seed_from_row(train.iloc[0])
        recency_scores_red = compute_recency_scores(train_num, range(1, 34), RED_COLS, half_life=60)
        recency_scores_blue = compute_recency_scores(
            train_num, range(1, 17), [BLUE_COL], half_life=40
        )
        red_stats = compute_red_stats(train_num)
        method_results = build_ssq_method_results(
            train_num, train_num.iloc[0], recency_scores_red, recency_scores_blue, red_stats, base_seed
        )

        if method_names is None:
            method_names = [item["name"] for item in method_results] + ["综合推荐(等权)"]
            stats = {name: {"red": [], "blue": [], "total": []} for name in method_names}

        target = df_sorted.iloc[idx]
        if pd.isna(target[BLUE_COL]):
            continue
        actual_reds = {int(target[col]) for col in RED_COLS if pd.notna(target[col])}
        actual_blue = int(target[BLUE_COL])

        for item in method_results:
            reds = {int(n) for n in item["reds"]}
            blue = int(item["blue"])
            red_hit = len(reds & actual_reds)
            blue_hit = 1 if blue == actual_blue else 0
            stats[item["name"]]["red"].append(red_hit)
            stats[item["name"]]["blue"].append(blue_hit)
            stats[item["name"]]["total"].append(red_hit + blue_hit)

        rng_reco = random.Random(base_seed + 97)
        ens_reds, ens_blue = build_ensemble_recommendation(
            method_results, recency_scores_red, recency_scores_blue, rng_reco
        )
        ens_red_hit = len(set(ens_reds) & actual_reds)
        ens_blue_hit = 1 if ens_blue == actual_blue else 0
        stats["综合推荐(等权)"]["red"].append(ens_red_hit)
        stats["综合推荐(等权)"]["blue"].append(ens_blue_hit)
        stats["综合推荐(等权)"]["total"].append(ens_red_hit + ens_blue_hit)

    if not stats:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for name, hits in stats.items():
        red_hits = hits["red"]
        if not red_hits:
            continue
        blue_hits = hits["blue"]
        total_hits = hits["total"]
        rows.append(
            {
                "方法": name,
                "回测期数": len(red_hits),
                "平均红球命中": round(float(np.mean(red_hits)), 3),
                "蓝球命中率": round(float(np.mean(blue_hits)), 3),
                "红球>=3命中率": round(float(np.mean([h >= 3 for h in red_hits])), 3),
                "平均总命中": round(float(np.mean(total_hits)), 3),
                "最高红球命中": int(max(red_hits)),
                "最高总命中": int(max(total_hits)),
            }
        )
    return pd.DataFrame(rows).sort_values("平均总命中", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def run_dlt_backtest(df: pd.DataFrame, backtest_periods: int, train_window: int) -> pd.DataFrame:
    # 大乐透回测统计
    df_sorted = df.sort_values("issue", ascending=False).reset_index(drop=True)
    max_periods = len(df_sorted) - train_window - 1
    periods = min(int(backtest_periods), int(max_periods))
    if periods <= 0:
        return pd.DataFrame()

    stats: Dict[str, Dict[str, List[int]]] = {}
    method_names: List[str] | None = None

    for idx in range(periods):
        train = df_sorted.iloc[idx + 1 : idx + 1 + train_window]
        if train.empty:
            continue
        train_num = to_numeric_dlt_df(train)
        if train_num.empty:
            continue

        base_seed = get_base_seed_from_row(train.iloc[0])
        recency_front = compute_recency_scores(train_num, range(1, 36), DLT_FRONT_COLS, half_life=60)
        recency_back = compute_recency_scores(train_num, range(1, 13), DLT_BACK_COLS, half_life=40)
        dlt_stats = compute_dlt_front_stats(train_num)
        method_results = build_dlt_method_results(
            train_num, train_num.iloc[0], recency_front, recency_back, dlt_stats, base_seed
        )

        if method_names is None:
            method_names = [item["name"] for item in method_results] + ["综合推荐(等权)"]
            stats = {name: {"front": [], "back": [], "total": []} for name in method_names}

        target = df_sorted.iloc[idx]
        if target[DLT_FRONT_COLS].isna().any() or target[DLT_BACK_COLS].isna().any():
            continue
        actual_fronts = {int(target[col]) for col in DLT_FRONT_COLS}
        actual_backs = {int(target[col]) for col in DLT_BACK_COLS}

        for item in method_results:
            fronts = {int(n) for n in item["fronts"]}
            backs = {int(n) for n in item["backs"]}
            front_hit = len(fronts & actual_fronts)
            back_hit = len(backs & actual_backs)
            stats[item["name"]]["front"].append(front_hit)
            stats[item["name"]]["back"].append(back_hit)
            stats[item["name"]]["total"].append(front_hit + back_hit)

        rng_reco = random.Random(base_seed + 97)
        ens_fronts, ens_backs = build_dlt_ensemble(
            method_results,
            recency_front,
            recency_back,
            dlt_stats["bucket_target"],
            rng_reco,
        )
        ens_front_hit = len(set(ens_fronts) & actual_fronts)
        ens_back_hit = len(set(ens_backs) & actual_backs)
        stats["综合推荐(等权)"]["front"].append(ens_front_hit)
        stats["综合推荐(等权)"]["back"].append(ens_back_hit)
        stats["综合推荐(等权)"]["total"].append(ens_front_hit + ens_back_hit)

    if not stats:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for name, hits in stats.items():
        front_hits = hits["front"]
        if not front_hits:
            continue
        back_hits = hits["back"]
        total_hits = hits["total"]
        rows.append(
            {
                "方法": name,
                "回测期数": len(front_hits),
                "平均前区命中": round(float(np.mean(front_hits)), 3),
                "平均后区命中": round(float(np.mean(back_hits)), 3),
                "前区>=3命中率": round(float(np.mean([h >= 3 for h in front_hits])), 3),
                "后区全中率": round(float(np.mean([h == 2 for h in back_hits])), 3),
                "平均总命中": round(float(np.mean(total_hits)), 3),
                "最高前区命中": int(max(front_hits)),
                "最高总命中": int(max(total_hits)),
            }
        )
    return pd.DataFrame(rows).sort_values("平均总命中", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def run_sd_backtest(df: pd.DataFrame, backtest_periods: int, train_window: int) -> pd.DataFrame:
    # 福彩3D 回测统计
    df_sorted = df.sort_values("issue", ascending=False).reset_index(drop=True)
    max_periods = len(df_sorted) - train_window - 1
    periods = min(int(backtest_periods), int(max_periods))
    if periods <= 0:
        return pd.DataFrame()

    stats: Dict[str, Dict[str, List[int]]] = {}
    method_names: List[str] | None = None

    for idx in range(periods):
        train = df_sorted.iloc[idx + 1 : idx + 1 + train_window]
        if train.empty:
            continue
        train_num = to_numeric_sd_df(train)
        if train_num.empty:
            continue

        base_seed = get_base_seed_from_row(train.iloc[0])
        sd_stats = compute_sd_stats(train_num)
        sd_recency_scores = compute_sd_recency_scores(train_num, half_life=60)
        method_results = build_sd_method_results(train_num, sd_stats, sd_recency_scores, base_seed)

        if method_names is None:
            method_names = [item["name"] for item in method_results] + ["综合推荐(等权)"]
            stats = {name: {"hit": []} for name in method_names}

        target = df_sorted.iloc[idx]
        if target[SD_DIGIT_COLS].isna().any():
            continue
        actual_digits = [int(target[col]) for col in SD_DIGIT_COLS]

        for item in method_results:
            digits = [int(d) for d in item["digits"]]
            hit = sum(
                1 for pos, val in enumerate(digits) if pos < len(actual_digits) and val == actual_digits[pos]
            )
            stats[item["name"]]["hit"].append(hit)

        rng_reco = random.Random(base_seed + 79)
        ens_digits = build_sd_ensemble(method_results, sd_recency_scores, rng_reco)
        ens_hit = sum(
            1 for pos, val in enumerate(ens_digits) if pos < len(actual_digits) and val == actual_digits[pos]
        )
        stats["综合推荐(等权)"]["hit"].append(ens_hit)

    if not stats:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for name, hits in stats.items():
        hit_list = hits["hit"]
        if not hit_list:
            continue
        rows.append(
            {
                "方法": name,
                "回测期数": len(hit_list),
                "平均定位命中": round(float(np.mean(hit_list)), 3),
                "二位及以上命中率": round(float(np.mean([h >= 2 for h in hit_list])), 3),
                "三位全中率": round(float(np.mean([h == 3 for h in hit_list])), 3),
                "最高定位命中": int(max(hit_list)),
            }
        )
    return pd.DataFrame(rows).sort_values("平均定位命中", ascending=False).reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="Lottery Visual Lab", layout="wide")

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.8rem; }
        .ball-row { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
        .ball { display: inline-block; padding: 6px 10px; border-radius: 999px; font-weight: 600; color: #fff; }
        .ball.red { background: #e74c3c; }
        .ball.blue { background: #3498db; }
        .ball.orange { background: #f39c12; }
        div[data-testid="stMetric"] { background: #ffffff; padding: 12px 16px; border-radius: 10px; border: 1px solid #f0f0f0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("彩票数据分析与趋势实验室")
    st.caption("提示：预测结果仅供娱乐与数学实验展示，不构成任何投注建议。")

    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.info(f"当前时间：{now_text}")

    with st.sidebar:
        st.header("控制面板")
        lottery = st.selectbox("彩种", ["双色球", "大乐透", "福彩3D"], index=0)

        if lottery == "双色球":
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
        elif lottery == "大乐透":
            data_file = st.text_input("数据文件路径", value=DLT_FILE_DEFAULT)
            data_path = Path(data_file)

            st.subheader("历史数据下载")
            limit = st.number_input("下载期数上限", min_value=100, max_value=10000, value=5000, step=100)
            if st.button("下载/更新历史数据"):
                if fetch_dlt_history is None:
                    st.error(f"无法调用抓取模块：{FETCH_DLT_IMPORT_ERROR}")
                else:
                    with st.spinner("正在下载并保存..."):
                        df_new = fetch_dlt_history(int(limit))
                        df_new.to_excel(data_path, index=False, engine="openpyxl")
                        st.cache_data.clear()
                    st.success(f"已保存：{data_path.resolve()}")

            st.subheader("分析期数")
            mode = st.radio("选择范围", ["最近100期", "自定义期数"], index=0)
            custom_n = st.number_input("自定义期数", min_value=20, max_value=10000, value=300, step=10)
        else:
            data_file = st.text_input("数据文件路径", value=SD_FILE_DEFAULT)
            data_path = Path(data_file)

            st.subheader("历史数据下载")
            limit = st.number_input("下载期数上限", min_value=100, max_value=10000, value=5000, step=100)
            if st.button("下载/更新历史数据"):
                if fetch_sd_history is None:
                    st.error(f"无法调用抓取模块：{FETCH_SD_IMPORT_ERROR}")
                else:
                    with st.spinner("正在下载并保存..."):
                        df_new = fetch_sd_history(int(limit))
                        df_new.to_excel(data_path, index=False, engine="openpyxl")
                        st.cache_data.clear()
                    st.success(f"已保存：{data_path.resolve()}")

            st.subheader("分析期数")
            mode = st.radio("选择范围", ["最近100期", "自定义期数"], index=0)
            custom_n = st.number_input("自定义期数", min_value=20, max_value=10000, value=300, step=10)

    if lottery == "福彩3D":
        if not data_path.exists():
            st.warning("未找到数据文件，请先下载历史数据。")
            return

        df_sd = load_sd_data(str(data_path))
        if df_sd.empty:
            st.warning("数据为空，请检查 Excel 文件内容。")
            return

        df_sd = df_sd.sort_values("issue", ascending=False).reset_index(drop=True)
        n_periods_sd = min(100, len(df_sd)) if mode == "最近100期" else min(int(custom_n), len(df_sd))

        df_sd_recent = df_sd.head(n_periods_sd).copy()
        df_sd_recent_num = to_numeric_sd_df(df_sd_recent)
        set_plot_context(data_path.name)

        latest_sd = df_sd.iloc[0]
        digits_latest = [latest_sd[col] for col in SD_DIGIT_COLS]
        sum_latest = int(pd.to_numeric(latest_sd[SD_SUM_COL], errors="coerce")) if pd.notna(
            latest_sd[SD_SUM_COL]
        ) else sum(int(d) for d in digits_latest)

        latest_issue_seed = int(str(latest_sd["issue"]).lstrip("0") or "0")
        latest_date_seed = (
            int(latest_sd[DATE_COL].strftime("%Y%m%d")) if pd.notna(latest_sd[DATE_COL]) else 0
        )
        base_seed = latest_issue_seed + latest_date_seed
        sd_stats = compute_sd_stats(df_sd_recent_num)
        sd_recency_scores = compute_sd_recency_scores(df_sd_recent_num, half_life=60)
        sd_method_weights: List[float] | None = None
        df_sd_prev = df_sd.iloc[1 : n_periods_sd + 1].copy()
        if not df_sd_prev.empty:
            df_sd_prev_num = to_numeric_sd_df(df_sd_prev)
            if not df_sd_prev_num.empty:
                latest_sd_prev = df_sd_prev.iloc[0]
                latest_prev_issue_seed = int(str(latest_sd_prev["issue"]).lstrip("0") or "0")
                latest_prev_date_seed = (
                    int(latest_sd_prev[DATE_COL].strftime("%Y%m%d"))
                    if pd.notna(latest_sd_prev[DATE_COL])
                    else 0
                )
                base_seed_prev = latest_prev_issue_seed + latest_prev_date_seed
                sd_stats_prev = compute_sd_stats(df_sd_prev_num)
                sd_recency_scores_prev = compute_sd_recency_scores(df_sd_prev_num, half_life=60)
                sd_prev_results = build_sd_method_results(
                    df_sd_prev_num, sd_stats_prev, sd_recency_scores_prev, base_seed_prev
                )
                actual_digits = [int(latest_sd[col]) for col in SD_DIGIT_COLS]
                sd_method_weights = compute_sd_method_weights(sd_prev_results, actual_digits)

        metric_cols = st.columns(4)
        metric_cols[0].metric("最新期号", latest_sd["issue"])
        metric_cols[1].metric(
            "最新日期", latest_sd[DATE_COL].strftime("%Y-%m-%d") if pd.notna(latest_sd[DATE_COL]) else "N/A"
        )
        metric_cols[2].metric("总期数", len(df_sd))
        metric_cols[3].metric("当前分析期数", n_periods_sd)

        st.subheader("最新一期号码")
        render_sd_row(digits_latest, sum_latest)

        tab_freq, tab_trend, tab_markov, tab_predict, tab_backtest, tab_data = st.tabs(
            ["频次分布", "趋势分析", "马尔科夫预测", "创意预测", "回测分析", "原始数据"]
        )

        with tab_freq:
            st.markdown("**位置与整体频次**")
            tag = f"sd_freq_{n_periods_sd}"
            path_pos = plot_sd_position_frequency(df_sd_recent_num, tag)
            path_all = plot_sd_overall_frequency(df_sd_recent_num, tag)
            st.caption(f"图片已保存：{path_pos}")
            st.caption(f"图片已保存：{path_all}")

        with tab_trend:
            st.markdown("**和值与奇偶趋势**")
            tag = f"sd_trend_{n_periods_sd}"
            path_sum = plot_sd_sum_trend(df_sd_recent_num, tag)
            path_dist = plot_sd_sum_distribution(df_sd_recent_num, tag)
            path_oe = plot_sd_odd_even_trend(df_sd_recent_num, tag)
            st.caption(f"图片已保存：{path_sum}")
            st.caption(f"图片已保存：{path_dist}")
            st.caption(f"图片已保存：{path_oe}")

        with tab_markov:
            render_sd_markov_tab(df_sd_recent_num)

        with tab_predict:
            st.markdown("**数学创意预测（仅供娱乐）**")
            rng_reco_base = random.Random(base_seed + 79)
            rng_reco_feedback = random.Random(base_seed + 79)

            sd_results = build_sd_method_results(
                df_sd_recent_num, sd_stats, sd_recency_scores, base_seed
            )
            sd_base = build_sd_ensemble(sd_results, sd_recency_scores, rng_reco_base)
            sd_base_sum = sum(sd_base)

            st.markdown("**基础推荐号码**")
            render_sd_row([str(d) for d in sd_base], sd_base_sum)
            st.code(format_sd_ticket(sd_base))
            st.caption("综合规则：多方法投票 + 近期热度微调。")

            if sd_method_weights:
                sd_feedback = build_sd_ensemble(
                    sd_results,
                    sd_recency_scores,
                    rng_reco_feedback,
                    method_weights=sd_method_weights,
                )
                sd_feedback_sum = sum(sd_feedback)
                st.markdown("**反馈推荐号码**")
                render_sd_row([str(d) for d in sd_feedback], sd_feedback_sum)
                st.code(format_sd_ticket(sd_feedback))
                st.caption("反馈逻辑：依据上一期实际号码对方法投票权重自动调整。")
            else:
                st.caption("反馈逻辑：历史期数不足，默认使用基础推荐。")

            for idx, item in enumerate(sd_results):
                expanded = idx == 0
                with st.expander(item["name"], expanded=expanded):
                    st.write(item["desc"])
                    st.code(format_sd_ticket(item["digits"]))

        with tab_backtest:
            st.markdown("**回测分析统计**")
            min_train = 30
            min_backtest = 10
            min_required = min_train + min_backtest + 1
            if len(df_sd) < min_required:
                st.warning(f"历史数据不足，回测至少需要 {min_required} 期。")
            else:
                max_train = len(df_sd) - min_backtest - 1
                train_window = st.number_input(
                    "训练窗口期数",
                    min_value=min_train,
                    max_value=max_train,
                    value=min(120, max_train),
                    step=10,
                    key="sd_backtest_train_window",
                )
                max_periods = len(df_sd) - int(train_window) - 1
                backtest_periods = st.number_input(
                    "回测期数",
                    min_value=min_backtest,
                    max_value=max_periods,
                    value=min(80, max_periods),
                    step=10,
                    key="sd_backtest_periods",
                )
                with st.spinner("正在回测统计..."):
                    backtest_df = run_sd_backtest(df_sd, int(backtest_periods), int(train_window))
                if backtest_df.empty:
                    st.info("回测结果为空，请检查数据的完整性。")
                else:
                    st.dataframe(backtest_df, use_container_width=True, height=420)
                st.caption("说明：回测使用滚动窗口模拟，命中率为 0-1 比例。")

        with tab_data:
            st.markdown("**最近数据预览**")
            st.dataframe(df_sd_recent, use_container_width=True, height=420)

        return

    if lottery == "大乐透":
        if not data_path.exists():
            st.warning("未找到数据文件，请先下载历史数据。")
            return

        df_dlt = load_dlt_data(str(data_path))
        if df_dlt.empty:
            st.warning("数据为空，请检查 Excel 文件内容。")
            return

        df_dlt = df_dlt.sort_values("issue", ascending=False).reset_index(drop=True)
        n_periods_dlt = min(100, len(df_dlt)) if mode == "最近100期" else min(int(custom_n), len(df_dlt))

        df_dlt_recent = df_dlt.head(n_periods_dlt).copy()
        df_dlt_recent_num = to_numeric_dlt_df(df_dlt_recent)
        set_plot_context(data_path.name)

        latest_dlt = df_dlt.iloc[0]
        front_latest = [latest_dlt[col] for col in DLT_FRONT_COLS]
        back_latest = [latest_dlt[col] for col in DLT_BACK_COLS]

        latest_issue_seed = int(str(latest_dlt["issue"]).lstrip("0") or "0")
        latest_date_seed = (
            int(latest_dlt[DATE_COL].strftime("%Y%m%d")) if pd.notna(latest_dlt[DATE_COL]) else 0
        )
        base_seed = latest_issue_seed + latest_date_seed
        recency_scores_front = compute_recency_scores(
            df_dlt_recent_num, range(1, 36), DLT_FRONT_COLS, half_life=60
        )
        recency_scores_back = compute_recency_scores(
            df_dlt_recent_num, range(1, 13), DLT_BACK_COLS, half_life=40
        )
        dlt_stats = compute_dlt_front_stats(df_dlt_recent_num)
        dlt_method_weights: List[float] | None = None
        df_dlt_prev = df_dlt.iloc[1 : n_periods_dlt + 1].copy()
        if not df_dlt_prev.empty:
            df_dlt_prev_num = to_numeric_dlt_df(df_dlt_prev)
            if not df_dlt_prev_num.empty:
                latest_dlt_prev = df_dlt_prev.iloc[0]
                latest_prev_issue_seed = int(str(latest_dlt_prev["issue"]).lstrip("0") or "0")
                latest_prev_date_seed = (
                    int(latest_dlt_prev[DATE_COL].strftime("%Y%m%d"))
                    if pd.notna(latest_dlt_prev[DATE_COL])
                    else 0
                )
                base_seed_prev = latest_prev_issue_seed + latest_prev_date_seed
                recency_front_prev = compute_recency_scores(
                    df_dlt_prev_num, range(1, 36), DLT_FRONT_COLS, half_life=60
                )
                recency_back_prev = compute_recency_scores(
                    df_dlt_prev_num, range(1, 13), DLT_BACK_COLS, half_life=40
                )
                dlt_stats_prev = compute_dlt_front_stats(df_dlt_prev_num)
                dlt_prev_results = build_dlt_method_results(
                    df_dlt_prev_num,
                    df_dlt_prev_num.iloc[0],
                    recency_front_prev,
                    recency_back_prev,
                    dlt_stats_prev,
                    base_seed_prev,
                )
                actual_fronts = [int(latest_dlt[col]) for col in DLT_FRONT_COLS]
                actual_backs = [int(latest_dlt[col]) for col in DLT_BACK_COLS]
                dlt_method_weights = compute_dlt_method_weights(
                    dlt_prev_results, actual_fronts, actual_backs
                )

        metric_cols = st.columns(4)
        metric_cols[0].metric("最新期号", latest_dlt["issue"])
        metric_cols[1].metric(
            "最新日期",
            latest_dlt[DATE_COL].strftime("%Y-%m-%d") if pd.notna(latest_dlt[DATE_COL]) else "N/A",
        )
        metric_cols[2].metric("总期数", len(df_dlt))
        metric_cols[3].metric("当前分析期数", n_periods_dlt)

        st.subheader("最新一期号码")
        render_dlt_row(front_latest, back_latest)

        tab_freq, tab_trend, tab_structure, tab_markov, tab_predict, tab_backtest, tab_data = st.tabs(
            ["频次分布", "号码走势", "结构趋势", "马尔科夫预测", "创意预测", "回测分析", "原始数据"]
        )

        with tab_freq:
            st.markdown("**频次统计（前区与后区）**")
            tag = f"dlt_freq_{n_periods_dlt}"
            path_front = plot_dlt_front_frequency(df_dlt_recent_num, tag)
            path_back = plot_dlt_back_frequency(df_dlt_recent_num, tag)
            st.caption(f"图片已保存：{path_front}")
            st.caption(f"图片已保存：{path_back}")

        with tab_trend:
            st.markdown("**走势散点与序列趋势**")
            tag = f"dlt_trend_{n_periods_dlt}"
            path_front = plot_dlt_front_trend(df_dlt_recent_num, tag)
            path_back = plot_dlt_back_trend(df_dlt_recent_num, tag)
            st.caption(f"图片已保存：{path_front}")
            st.caption(f"图片已保存：{path_back}")

        with tab_structure:
            st.markdown("**和值、跨度与奇偶结构**")
            tag = f"dlt_struct_{n_periods_dlt}"
            path_sum = plot_dlt_sum_trend(df_dlt_recent_num, tag)
            path_span = plot_dlt_span_trend(df_dlt_recent_num, tag)
            path_oe = plot_dlt_odd_even_trend(df_dlt_recent_num, tag)
            st.caption(f"图片已保存：{path_sum}")
            st.caption(f"图片已保存：{path_span}")
            st.caption(f"图片已保存：{path_oe}")

        with tab_markov:
            render_dlt_markov_tab(df_dlt_recent_num)

        with tab_predict:
            st.markdown("**数学创意预测（仅供娱乐）**")
            rng_reco_base = random.Random(base_seed + 97)
            rng_reco_feedback = random.Random(base_seed + 97)

            dlt_results = build_dlt_method_results(
                df_dlt_recent_num,
                df_dlt_recent_num.iloc[0],
                recency_scores_front,
                recency_scores_back,
                dlt_stats,
                base_seed,
            )

            base_fronts, base_backs = build_dlt_ensemble(
                dlt_results,
                recency_scores_front,
                recency_scores_back,
                dlt_stats["bucket_target"],
                rng_reco_base,
            )

            st.markdown("**基础推荐号码**")
            render_dlt_row([f"{n:02d}" for n in base_fronts], [f"{n:02d}" for n in base_backs])
            st.code(format_dlt_ticket(base_fronts, base_backs))
            st.caption("综合规则：多方法投票 + 近期热度微调 + 分区约束。")

            if dlt_method_weights:
                feedback_fronts, feedback_backs = build_dlt_ensemble(
                    dlt_results,
                    recency_scores_front,
                    recency_scores_back,
                    dlt_stats["bucket_target"],
                    rng_reco_feedback,
                    method_weights=dlt_method_weights,
                )
                st.markdown("**反馈推荐号码**")
                render_dlt_row(
                    [f"{n:02d}" for n in feedback_fronts], [f"{n:02d}" for n in feedback_backs]
                )
                st.code(format_dlt_ticket(feedback_fronts, feedback_backs))
                st.caption("反馈逻辑：依据上一期实际号码对方法投票权重自动调整。")
            else:
                st.caption("反馈逻辑：历史期数不足，默认使用基础推荐。")

            for idx, item in enumerate(dlt_results):
                expanded = idx == 0
                with st.expander(item["name"], expanded=expanded):
                    st.write(item["desc"])
                    st.code(format_dlt_ticket(item["fronts"], item["backs"]))

        with tab_backtest:
            st.markdown("**回测分析统计**")
            min_train = 30
            min_backtest = 10
            min_required = min_train + min_backtest + 1
            if len(df_dlt) < min_required:
                st.warning(f"历史数据不足，回测至少需要 {min_required} 期。")
            else:
                max_train = len(df_dlt) - min_backtest - 1
                train_window = st.number_input(
                    "训练窗口期数",
                    min_value=min_train,
                    max_value=max_train,
                    value=min(120, max_train),
                    step=10,
                    key="dlt_backtest_train_window",
                )
                max_periods = len(df_dlt) - int(train_window) - 1
                backtest_periods = st.number_input(
                    "回测期数",
                    min_value=min_backtest,
                    max_value=max_periods,
                    value=min(80, max_periods),
                    step=10,
                    key="dlt_backtest_periods",
                )
                with st.spinner("正在回测统计..."):
                    backtest_df = run_dlt_backtest(df_dlt, int(backtest_periods), int(train_window))
                if backtest_df.empty:
                    st.info("回测结果为空，请检查数据的完整性。")
                else:
                    st.dataframe(backtest_df, use_container_width=True, height=420)
                st.caption("说明：回测使用滚动窗口模拟，命中率为 0-1 比例。")

        with tab_data:
            st.markdown("**最近数据预览**")
            st.dataframe(df_dlt_recent, use_container_width=True, height=420)

        return

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
    set_plot_context(data_path.name)

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
    ssq_method_weights: List[float] | None = None
    df_prev = df.iloc[1 : n_periods + 1].copy()
    if not df_prev.empty:
        df_prev_num = to_numeric_df(df_prev)
        if not df_prev_num.empty:
            latest_prev = df_prev.iloc[0]
            latest_prev_issue_seed = int(str(latest_prev["issue"]).lstrip("0") or "0")
            latest_prev_date_seed = (
                int(latest_prev[DATE_COL].strftime("%Y%m%d")) if pd.notna(latest_prev[DATE_COL]) else 0
            )
            base_seed_prev = latest_prev_issue_seed + latest_prev_date_seed
            recency_red_prev = compute_recency_scores(
                df_prev_num, range(1, 34), RED_COLS, half_life=60
            )
            recency_blue_prev = compute_recency_scores(
                df_prev_num, range(1, 17), [BLUE_COL], half_life=40
            )
            red_stats_prev = compute_red_stats(df_prev_num)
            ssq_prev_results = build_ssq_method_results(
                df_prev_num,
                df_prev_num.iloc[0],
                recency_red_prev,
                recency_blue_prev,
                red_stats_prev,
                base_seed_prev,
            )
            actual_reds = [int(latest[col]) for col in RED_COLS]
            actual_blue = int(latest[BLUE_COL])
            ssq_method_weights = compute_ssq_method_weights(
                ssq_prev_results, actual_reds, actual_blue
            )

    metric_cols = st.columns(4)
    metric_cols[0].metric("最新期号", latest["issue"])
    metric_cols[1].metric("最新日期", latest[DATE_COL].strftime("%Y-%m-%d") if pd.notna(latest[DATE_COL]) else "N/A")
    metric_cols[2].metric("总期数", len(df))
    metric_cols[3].metric("当前分析期数", n_periods)

    st.subheader("最新一期号码")
    render_ball_row(red_latest, blue_latest)

    tab_freq, tab_trend, tab_structure, tab_markov, tab_predict, tab_backtest, tab_data = st.tabs(
        ["频次分布", "号码走势", "结构趋势", "马尔科夫预测", "创意预测", "回测分析", "原始数据"]
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

    with tab_markov:
        render_ssq_markov_tab(df_recent_num)

    with tab_predict:
        st.markdown("**数学创意预测（仅供娱乐）**")
        rng_reco_base = random.Random(base_seed + 97)
        rng_reco_feedback = random.Random(base_seed + 97)

        method_results = build_ssq_method_results(
            df_recent_num, latest_num, recency_scores_red, recency_scores_blue, red_stats, base_seed
        )

        base_reds, base_blue = build_ensemble_recommendation(
            method_results,
            recency_scores_red,
            recency_scores_blue,
            rng_reco_base,
        )

        st.markdown("**基础推荐号码**")
        render_ball_row([f"{n:02d}" for n in base_reds], f"{base_blue:02d}")
        st.code(format_ticket(base_reds, base_blue))
        st.caption("综合规则：多方法投票 + 近期热度微调 + 分区约束。")

        if ssq_method_weights:
            feedback_reds, feedback_blue = build_ensemble_recommendation(
                method_results,
                recency_scores_red,
                recency_scores_blue,
                rng_reco_feedback,
                method_weights=ssq_method_weights,
            )
            st.markdown("**反馈推荐号码**")
            render_ball_row([f"{n:02d}" for n in feedback_reds], f"{feedback_blue:02d}")
            st.code(format_ticket(feedback_reds, feedback_blue))
            st.caption("反馈逻辑：依据上一期实际号码对方法投票权重自动调整。")
        else:
            st.caption("反馈逻辑：历史期数不足，默认使用基础推荐。")

        for idx, item in enumerate(method_results):
            expanded = idx == 0
            with st.expander(item["name"], expanded=expanded):
                st.write(item["desc"])
                st.code(format_ticket(item["reds"], int(item["blue"])))

    with tab_backtest:
        st.markdown("**回测分析统计**")
        min_train = 30
        min_backtest = 10
        min_required = min_train + min_backtest + 1
        if len(df) < min_required:
            st.warning(f"历史数据不足，回测至少需要 {min_required} 期。")
        else:
            max_train = len(df) - min_backtest - 1
            train_window = st.number_input(
                "训练窗口期数",
                min_value=min_train,
                max_value=max_train,
                value=min(120, max_train),
                step=10,
                key="ssq_backtest_train_window",
            )
            max_periods = len(df) - int(train_window) - 1
            backtest_periods = st.number_input(
                "回测期数",
                min_value=min_backtest,
                max_value=max_periods,
                value=min(80, max_periods),
                step=10,
                key="ssq_backtest_periods",
            )
            with st.spinner("正在回测统计..."):
                backtest_df = run_ssq_backtest(df, int(backtest_periods), int(train_window))
            if backtest_df.empty:
                st.info("回测结果为空，请检查数据的完整性。")
            else:
                st.dataframe(backtest_df, use_container_width=True, height=420)
            st.caption("说明：回测使用滚动窗口模拟，命中率为 0-1 比例。")

    with tab_data:
        st.markdown("**最近数据预览**")
        st.dataframe(df_recent, use_container_width=True, height=420)


if __name__ == "__main__":
    main()
