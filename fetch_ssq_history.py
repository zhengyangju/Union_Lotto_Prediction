# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


DEFAULT_URL = "https://datachart.500star.com/ssq/history/newinc/history.php"


def fetch_ssq_history(limit: int) -> pd.DataFrame:
    # 使用 GBK 解码网页，避免数据表解析乱码
    response = requests.get(DEFAULT_URL, params={"limit": limit, "sort": "desc"}, timeout=30)
    response.encoding = "gbk"

    tables = pd.read_html(StringIO(response.text), attrs={"id": "tablelist"})
    if not tables:
        raise RuntimeError("未找到双色球历史表格数据")

    df = tables[0]
    expected_cols = 16
    if df.shape[1] != expected_cols:
        raise RuntimeError(f"表格列数异常：{df.shape[1]} != {expected_cols}")

    df.columns = [
        "issue",
        "red_1",
        "red_2",
        "red_3",
        "red_4",
        "red_5",
        "red_6",
        "blue",
        "blue2",
        "sales",
        "first_prize_count",
        "first_prize_amount",
        "second_prize_count",
        "second_prize_amount",
        "jackpot",
        "draw_date",
    ]

    # 号码列先转数值并过滤无效行
    ball_cols = ["red_1", "red_2", "red_3", "red_4", "red_5", "red_6", "blue"]
    df["issue"] = pd.to_numeric(df["issue"], errors="coerce")
    for col in ball_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["issue"] + ball_cols).copy()

    # 统一格式，避免前导零丢失
    df["issue"] = df["issue"].astype(int).astype(str).str.zfill(5)
    for col in ball_cols:
        df[col] = df[col].astype(int).map(lambda x: f"{x:02d}")

    # 删除不需要的列
    drop_cols = [
        "blue2",
        "sales",
        "first_prize_count",
        "first_prize_amount",
        "second_prize_count",
        "second_prize_amount",
        "jackpot",
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # 固定输出列顺序
    keep_cols = ["issue", "red_1", "red_2", "red_3", "red_4", "red_5", "red_6", "blue", "draw_date"]
    df = df[keep_cols].reset_index(drop=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="获取双色球历史数据并保存为 Excel。")
    parser.add_argument("--limit", type=int, default=5000, help="抓取期数上限，默认 5000。")
    parser.add_argument(
        "--output",
        type=str,
        default="ssq_history.xlsx",
        help="输出 Excel 文件路径，默认 ssq_history.xlsx。",
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    data = fetch_ssq_history(args.limit)
    data.to_excel(output_path, index=False, engine="openpyxl")
    print(f"已保存：{output_path}")


if __name__ == "__main__":
    main()
