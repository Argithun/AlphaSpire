"""
scripts/annotate_fields_with_categories.py

读取 data/wq_field/fields.csv（或项目根的 fields.csv），按规则把 field 归类为 final_category，
并输出:
  - data/wq_field/fields_typed.csv
  - data/wq_field/fields_typed.json
  - data/wq_field/fields_review.csv  (UNKNOWN 或 Other 的清单，供人工核对)

用法:
  pip install pandas
  python scripts/annotate_fields_with_categories.py --in path/to/fields.csv

注:
  - 脚本会自动检测常见列名（id/name/description/dataset/type）。
  - CATEGORY_MAP 可按需调整，优先基于 dataset 决定类；其次用字段名/描述关键词匹配。
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# 输出目录
BASE = Path(__file__).resolve().parents[1]
FIELD_DIR = BASE / "data" / "wq_fields"
DEFAULT_IN = FIELD_DIR / "fields.csv"

TEMP_DIR = BASE / "data" / "wq_template_fields"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = TEMP_DIR / "template_fields.csv"
OUT_JSON = TEMP_DIR / "template_fields.json"
REVIEW_CSV = TEMP_DIR / "review.csv"

# ----- 分类配置（可编辑） -----
# 每个类别：可包含 "datasets"（优先匹配）与 keywords（在 name/description 中匹配）
# keywords 使用小写，做 substring 检查；也可用正则（在后面扩展）
CATEGORY_MAP = {
    "Price": {
        "datasets": ["pv1"],
        "keywords": ["close", "open", "high", "low", "vwap", "last", "price", "midprice"]
    },
    "Volume": {
        "datasets": ["pv1"],
        "keywords": ["volume", "turnover", "vol", "sharestraded", "adv", "avgvol"]
    },
    "Returns": {
        "datasets": ["pv1"],
        "keywords": ["return", "pct", "pct_change", "ret", "logreturn", "return_"]
    },
    "MarketCap_Shares": {
        "datasets": ["pv1", "fundamental6"],
        "keywords": ["marketcap", "market cap", "cap", "shares", "sharesout", "float"]
    },
    "Identifiers_Metadata": {
        "datasets": [],
        "keywords": ["exchange", "ticker", "isin", "sedol", "cusip", "country", "currency", "symbol"]
    },
    "Corporate_Actions": {
        "datasets": ["fundamental6"],
        "keywords": ["dividend", "split", "exdiv", "spin-off", "merger", "delist"]
    },
    "Classification_Group": {
        "datasets": [],
        "keywords": ["industry", "sector", "gics", "subindustry", "indclass", "classification", "group_id", "group"]
    },
    "News_Sentiment": {
        "datasets": ["news12"],
        "keywords": ["news", "sentiment", "headline", "mention", "article", "buzz"]
    },
    "Analyst_Estimate": {
        "datasets": ["analyst4"],
        "keywords": ["estimate", "eps_forecast", "analyst", "consensus", "target_price", "revision"]
    },
    "Model_Score": {
        "datasets": ["model16"],
        "keywords": ["model", "score", "signal", "model_output", "alpha_model"]
    },
    "Fundamental_Income": {
        "datasets": ["fundamental6"],
        "keywords": ["revenue", "sales", "eps", "net_income", "profit", "gross_income", "operating_income"]
    },
    "Fundamental_Balance": {
        "datasets": ["fundamental6"],
        "keywords": ["asset", "liabilit", "equity", "book_value", "total_assets", "intangible", "cash_and"]
    },
    "Fundamental_Cashflow": {
        "datasets": ["fundamental6"],
        "keywords": ["cashflow", "operating_cash", "free_cash", "fcf", "cash_flow"]
    },
    "Fundamental_Ratio": {
        "datasets": ["fundamental6"],
        "keywords": ["pe", "pb", "roe", "roa", "margin", "ratio", "yield", "turnover_ratio"]
    },
    "Fundamental_Events": {
        "datasets": ["fundamental6"],
        "keywords": ["earnings_date", "earnings", "filing", "announcement", "event", "ipo", "merger", "acquisition"]
    },
    "Technical_Indicator": {
        "datasets": ["pv1"],
        "keywords": ["sma", "ema", "rsi", "macd", "bollinger", "adx", "roc", "stochastic"]
    },
    "TimeSeries_Feature": {
        "datasets": [],
        "keywords": ["lag", "diff", "rolling", "ts_", "window", "lookback", "momentum", "decay", "zscore"]
    },
    "Group_VECTOR": {
        "datasets": [],
        "keywords": ["bucket", "group_index", "groupid", "group_id", "category_index", "cartesian", "bucket_index"]
    },
    # Catch-all categories (less preferred)
    "Fundamental_Other": {
        "datasets": ["fundamental6"],
        "keywords": []
    },
    "Other": {
        "datasets": [],
        "keywords": []
    }
}
# -------------------------------

# helper: flatten keywords map for fast lookup
def _lower(x: Any) -> str:
    return "" if x is None else str(x).lower()

def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Return mapping of expected cols: id_col, name_col, desc_col, dataset_col, type_col"""
    cols = {c.lower(): c for c in df.columns}
    id_col = next((cols[k] for k in ("id","field_id","field","name","fieldname") if k in cols), None)
    name_col = next((cols[k] for k in ("name","field_name","display_name") if k in cols), id_col)
    desc_col = next((cols[k] for k in ("description","desc","long_description","details") if k in cols), None)
    dataset_col = next((cols[k] for k in ("dataset","dataset_id","source_dataset") if k in cols), None)
    type_col = next((cols[k] for k in ("type","data_type","field_type") if k in cols), None)
    return {"id": id_col, "name": name_col, "desc": desc_col, "dataset": dataset_col, "type": type_col}

# helper for matching quality
def best_keyword_match(text: str, keywords: List[str]) -> int:
    """
    返回在 text 中匹配到的关键词数量（越多越好）。
    keywords 已经是小写的关键词列表；text 已经小写。
    """
    if not keywords:
        return 0
    count = 0
    for kw in keywords:
        if not kw:
            continue
        if kw in text:
            count += 1
    return count

# 新的优先级列表（更具体的 category 放在前面）
CATEGORY_PRIORITY = [
    "News_Sentiment", "Analyst_Estimate", "Model_Score",
    "Fundamental_Income", "Fundamental_Balance", "Fundamental_Cashflow", "Fundamental_Ratio", "Fundamental_Events",
    "Technical_Indicator", "Price", "Volume", "MarketCap_Shares",
    "Classification_Group", "Identifiers_Metadata", "Group_VECTOR",
    "TimeSeries_Feature", "Corporate_Actions",
    "Fundamental_Other", "Other"
]

def classify_row(row: Dict[str, Any], col_map: Dict[str,str]) -> str:
    """
    改进后的分类逻辑：
      1) 先根据 dataset 找 candidate categories（但不立刻返回）
      2) 在 candidate 中按关键词匹配打分（匹配关键词数量越多优先）
      3) 若 candidate 没有关键词匹配，则在全局 categories 中按关键词匹配打分并选最优
      4) 若仍无匹配，使用 type 回退规则
      5) 最终返回 Other
    """
    name = _lower(row.get(col_map["name"], "")) if col_map["name"] else ""
    desc = _lower(row.get(col_map["desc"], "")) if col_map["desc"] else ""
    dataset = _lower(row.get(col_map["dataset"], "")) if col_map["dataset"] else ""
    dtype = _lower(row.get(col_map["type"], "")) if col_map["type"] else ""
    text = (name + " " + desc).strip()

    # 1) 找出 dataset 命中的候选 categories（保持优先级顺序）
    dataset_candidates = []
    for cat in CATEGORY_PRIORITY:
        meta = CATEGORY_MAP.get(cat, {})
        ds_list = [d.lower() for d in meta.get("datasets", [])]
        for ds in ds_list:
            if ds and ds in dataset:
                dataset_candidates.append(cat)
                break

    # 2) 在 dataset_candidates 中按关键词计分并选择最优
    best_cat = None
    best_score = 0
    for cat in dataset_candidates:
        kws = [k.lower() for k in CATEGORY_MAP.get(cat, {}).get("keywords", [])]
        score = best_keyword_match(text, kws)
        if score > best_score:
            best_score = score
            best_cat = cat

    if best_cat and best_score > 0:
        return best_cat

    # 3) 如果 dataset_candidates 存在但没有关键词匹配（即 dataset 很通用），
    #    我们选择在 dataset_candidates 中返回第一个（按 CATEGORY_PRIORITY），
    #    但仅当 dataset_candidates 的长度为1 或者我们允许 dataset-only 映射时。
    #    为避免把所有 pv1 都投到 Price，可以设阈值：如果 dataset_candidates 有多个，
    #    不自动选，转入全局 keyword 匹配；如果只有一个，则选它。
    if len(dataset_candidates) == 1:
        return dataset_candidates[0]

    # 4) 全局关键词匹配（按 priority 遍历，找到第一个匹配多个关键词的 category）
    best_cat = None
    best_score = 0
    for cat in CATEGORY_PRIORITY:
        kws = [k.lower() for k in CATEGORY_MAP.get(cat, {}).get("keywords", [])]
        score = best_keyword_match(text, kws)
        if score > best_score:
            best_score = score
            best_cat = cat

    if best_cat and best_score > 0:
        return best_cat

    # 5) 回退：根据 dtype hints
    if "group" in dtype:
        return "Group_VECTOR"
    if "symbol" in dtype or "id" in dtype or "string" in dtype:
        return "Identifiers_Metadata"
    if "matrix" in dtype or "time" in dtype or "vector" in dtype:
        if dataset and "fundamental" in dataset:
            return "Fundamental_Other"
        return "TimeSeries_Feature"

    # 6) 默认返回 Other
    return "Other"

def annotate(df: pd.DataFrame) -> pd.DataFrame:
    col_map = detect_columns(df)
    logging.info(f"Detected columns map: {col_map}")
    # ensure columns exist
    output_rows = []
    for _, r in df.iterrows():
        row = r.to_dict()
        cat = classify_row(row, col_map)
        row_out = dict(row)
        row_out["final_category"] = cat
        # keep normalized fields for convenience
        row_out["_detected_dataset"] = row.get(col_map["dataset"]) if col_map["dataset"] else ""
        row_out["_detected_type"] = row.get(col_map["type"]) if col_map["type"] else ""
        output_rows.append(row_out)
    return pd.DataFrame(output_rows)

def generate_template_fields():
    if not DEFAULT_IN.exists():
        logging.error(f"Input file not found: {DEFAULT_IN}")
        return

    logging.info(f"Loading {DEFAULT_IN}")
    df = pd.read_csv(DEFAULT_IN, dtype=str, keep_default_na=False)
    annotated = annotate(df)

    # write CSV / JSON
    annotated.to_csv(OUT_CSV, index=False, encoding="utf-8")
    records = annotated.to_dict(orient="records")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # generate review list: UNKNOWN or Other or categories flagged for review
    review_df = annotated[annotated["final_category"].isin(["Other", "Fundamental_Other"])]
    review_df.to_csv(REVIEW_CSV, index=False, encoding="utf-8")
    logging.info(f"Wrote {OUT_CSV}, {OUT_JSON}, review file {REVIEW_CSV}")
    logging.info(f"Total fields: {len(df)}, annotated: {len(records)}, need review: {len(review_df)}")


if __name__ == "__main__":
    generate_template_fields()
