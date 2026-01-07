# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


DEFAULT_URL = "https://datachart.500star.com/sd/history/inc/history.php"


def fetch_sd_history(limit: int | None = None) -> pd.DataFrame:
    # 使用 GBK 解码网页，避免历史表乱码
    params = {"limit": int(limit)} if limit else None
    response = requests.get(DEFAULT_URL, params=params, timeout=30)
    response.encoding = "gbk"

    tables = pd.read_html(StringIO(response.text), attrs={"id": "tablelist"})
    if not tables:
        raise RuntimeError("未找到福彩3D历史表格数据")

    df = tables[0]
    df = df[df[0] != "期号"].copy()
    if df.empty or df.shape[1] < 11:
        raise RuntimeError("福彩3D历史表结构异常")

    df.columns = [
        "issue",
        "numbers",
        "sum",
        "sales",
        "direct_count",
        "direct_amount",
        "group3_count",
        "group3_amount",
        "group6_count",
        "group6_amount",
        "draw_date",
    ]

    number_text = df["numbers"].astype(str).str.replace(r"[^0-9]", " ", regex=True).str.strip()
    digits = number_text.str.split(r"\s+", expand=True)
    if digits.shape[1] < 3:
        raise RuntimeError("中奖号码解析失败")

    df["d1"] = digits[0]
    df["d2"] = digits[1]
    df["d3"] = digits[2]

    df["issue"] = pd.to_numeric(df["issue"], errors="coerce")
    for col in ["d1", "d2", "d3", "sum"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["issue", "d1", "d2", "d3", "sum"]).copy()

    df["issue"] = df["issue"].astype(int).astype(str)
    for col in ["d1", "d2", "d3"]:
        df[col] = df[col].astype(int).astype(str)
    df["sum"] = df["sum"].astype(int)

    df = df[["issue", "d1", "d2", "d3", "sum", "draw_date"]]

    if limit:
        df = df.head(int(limit)).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="获取福彩3D历史数据并保存为 Excel。")
    parser.add_argument("--limit", type=int, default=5000, help="抓取期数上限，默认 5000。")
    parser.add_argument(
        "--output",
        type=str,
        default="sd_history.xlsx",
        help="输出 Excel 文件路径，默认 sd_history.xlsx。",
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    data = fetch_sd_history(args.limit)
    data.to_excel(output_path, index=False, engine="openpyxl")
    print(f"已保存：{output_path}")


if __name__ == "__main__":
    main()
