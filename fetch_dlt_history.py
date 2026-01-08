# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


DEFAULT_URL = "https://datachart.500.com/dlt/history/newinc/history.php"


def fetch_dlt_history(limit: int) -> pd.DataFrame:
    # 使用 GBK 解码网页，避免历史表乱码
    response = requests.get(DEFAULT_URL, params={"limit": limit, "sort": "desc"}, timeout=30)
    response.encoding = "gbk"

    tables = pd.read_html(StringIO(response.text), attrs={"id": "tablelist"})
    if not tables:
        raise RuntimeError("未找到大乐透历史表格数据")

    df = tables[0]
    expected_cols = 15
    if df.shape[1] != expected_cols:
        raise RuntimeError(f"表格列数异常：{df.shape[1]} != {expected_cols}")

    df.columns = [
        "issue",
        "front_1",
        "front_2",
        "front_3",
        "front_4",
        "front_5",
        "back_1",
        "back_2",
        "sales",
        "first_prize_count",
        "first_prize_amount",
        "second_prize_count",
        "second_prize_amount",
        "jackpot",
        "draw_date",
    ]

    ball_cols = ["front_1", "front_2", "front_3", "front_4", "front_5", "back_1", "back_2"]
    df["issue"] = pd.to_numeric(df["issue"], errors="coerce")
    for col in ball_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["issue"] + ball_cols).copy()

    df["issue"] = df["issue"].astype(int).astype(str).str.zfill(5)
    for col in ball_cols:
        df[col] = df[col].astype(int).map(lambda x: f"{x:02d}")

    drop_cols = [
        "sales",
        "first_prize_count",
        "first_prize_amount",
        "second_prize_count",
        "second_prize_amount",
        "jackpot",
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    keep_cols = [
        "issue",
        "front_1",
        "front_2",
        "front_3",
        "front_4",
        "front_5",
        "back_1",
        "back_2",
        "draw_date",
    ]
    df = df[keep_cols].reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="获取大乐透历史数据并保存为 Excel。")
    parser.add_argument("--limit", type=int, default=5000, help="抓取期数上限，默认 5000。")
    parser.add_argument(
        "--output",
        type=str,
        default="dlt_history.xlsx",
        help="输出 Excel 文件路径，默认 dlt_history.xlsx。",
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    data = fetch_dlt_history(args.limit)
    data.to_excel(output_path, index=False, engine="openpyxl")
    print(f"已保存：{output_path}")

##
if __name__ == "__main__":
    main()
