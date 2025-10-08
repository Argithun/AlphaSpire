import json
import csv
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from openai import OpenAI
import hdbscan
import warnings
warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'",
    category=FutureWarning,
)

from utils.config_loader import ConfigLoader

BASE_DIR = Path(__file__).resolve().parents[1]
FIELDS_DIR = BASE_DIR / "data" / "wq_fields"
OUTPUT_DIR = BASE_DIR / "data" / "wq_template_fields"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUTPUT_DIR / "template_fields.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# =========================
# Step 1. 加载所有 csv 文件
# =========================
def load_all_fields() -> pd.DataFrame:
    """读取 data/wq_fields 下所有有效 CSV 文件并合并"""
    dfs = []
    for file in FIELDS_DIR.glob("*.csv"):
        if file.stat().st_size == 0:
            logging.warning(f"⚠️ Skipping empty file (0 bytes): {file.name}")
            continue
        try:
            df = pd.read_csv(file, dtype=str, keep_default_na=False)
            if df.empty or len(df.columns) == 0:
                logging.warning(f"⚠️ Skipping file with no valid columns: {file.name}")
                continue
            df["__dataset__"] = file.stem
            dfs.append(df)

        except pd.errors.EmptyDataError:
            logging.warning(f"⚠️ Skipping malformed file (EmptyDataError): {file.name}")
            continue
        except pd.errors.ParserError as e:
            logging.warning(f"⚠️ Skipping broken CSV (ParserError): {file.name} ({e})")
            continue
        except Exception as e:
            logging.error(f"❌ Unexpected error reading {file.name}: {e}")
            continue
    if not dfs:
        raise RuntimeError(f"❌ No valid CSV files found in {FIELDS_DIR}")

    return pd.concat(dfs, ignore_index=True)


# =========================
# Step 2. 聚类逻辑
# =========================
def cluster_fields_by_semantics_auto(df: pd.DataFrame,
                                     min_cluster_size: int = 3,
                                     min_samples: int = 2) -> Dict[int, List[str]]:
    """
    使用 HDBSCAN 自动确定聚类数量的语义聚类方法。
    基于 id + description 文本表示。
    """
    if len(df) <= min_cluster_size:
        # 数据太少，不聚类
        return {0: df["id"].tolist()}

    texts = (df["id"].astype(str) + " " + df["description"].astype(str)).tolist()

    # Step 1. TF-IDF 向量化
    tfidf = TfidfVectorizer(max_features=2000)
    X = tfidf.fit_transform(texts)

    # Step 2. HDBSCAN 聚类
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(X.toarray())

    # Step 3. 聚类结果收集
    clusters: Dict[int, List[str]] = {}
    for idx, label in enumerate(labels):
        if label == -1:
            # -1 表示噪声，可选择丢弃或单独归类
            label = 9999  # 归入“噪声”类
        clusters.setdefault(label, []).append(df.iloc[idx]["id"])

    # 可选：按簇大小排序
    clusters = dict(sorted(clusters.items(), key=lambda x: -len(x[1])))
    return clusters


# =========================
# Step 3. 调用 LLM 命名类别
# =========================
def get_llm_client():
    return OpenAI(
        base_url=ConfigLoader.get("openai_base_url"),
        api_key=ConfigLoader.get("openai_api_key"),
    )


def name_cluster_with_llm(client, type_name: str, dataset: str, sample_texts: List[str]) -> str:
    """调用 LLM 生成聚类名称"""
    joined = "\n".join(sample_texts[:5])  # 只取前几个字段描述
    prompt = f"""
You are classifying quantitative finance data fields.
Given the dataset = {dataset} and field type = {type_name}.
Below are some field examples:
{joined}

Please propose a short, meaningful lowercase name (1-3 words) for this group, 
like "momentum", "valuation_ratio", "sentiment_score", etc.
Return only the name.
"""
    resp = client.chat.completions.create(
        model=ConfigLoader.get("openai_model_name"),
        messages=[{"role": "system", "content": "You are a finance data classifier."},
                  {"role": "user", "content": prompt}],
        temperature=0.3,
    )
    name = resp.choices[0].message.content.strip()
    # 清理非法字符
    name = name.replace(" ", "_").replace("-", "_").lower()
    return name


# =========================
# Step 4. 主生成逻辑
# =========================
def generate_template_fields_v2():
    logging.info("📥 Loading all field csvs...")
    df = load_all_fields()

    # 自动检测必要列
    expected_cols = {"id", "description", "type"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns in input: {expected_cols - set(df.columns)}")

    client = get_llm_client()
    all_mappings = {}

    # 按 dataset + type 分组
    grouped = df.groupby(["__dataset__", "type"])
    for (dataset, dtype), subdf in grouped:
        if len(subdf) < 3:
            print(f"Skipping small group: {dataset}:{dtype} ({len(subdf)})")
            continue

        print(f"🧩 Processing dataset={dataset}, type={dtype}, size={len(subdf)}")
        clusters = cluster_fields_by_semantics_auto(subdf)

        for cluster_id, field_ids in clusters.items():
            sample_df = subdf[subdf["id"].isin(field_ids)]
            sample_texts = (sample_df["id"] + " " + sample_df["description"]).tolist()
            type_name = name_cluster_with_llm(client, dtype, dataset, sample_texts)

            key = f"</{type_name}:{dtype}:{dataset}/>"
            all_mappings[key] = field_ids
            print(f"✅ Generated type: {key} ({len(field_ids)} fields)")

    # 保存结果
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_mappings, f, ensure_ascii=False, indent=2)
    print(f"🎯 Saved {len(all_mappings)} template field types to {OUT_JSON}")


if __name__ == "__main__":
    generate_template_fields_v2()
