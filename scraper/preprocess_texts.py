import os
import json
import unicodedata
from pathlib import Path
from bs4 import BeautifulSoup
from loguru import logger

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "wq_posts" / "raw_posts"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = BASE_DIR / "data" / "wq_posts" / "processed_posts"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """
    清洗文本中非utf-8字符，统一为 NFC 格式
    """
    if not text:
        return ""
    # 先 encode/decode 丢弃非法字符
    cleaned = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    # 再做 Unicode 归一化，避免奇怪的变体
    cleaned = unicodedata.normalize("NFC", cleaned)
    return cleaned.strip()

def extract_post_info(html_content: str) -> dict:
    """从单个HTML中抽取 description, title, post-body, post-comments"""
    soup = BeautifulSoup(html_content, "html.parser")

    # description
    description = ""
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        description = meta_desc.get("content").strip()
    if not description:  # 备用
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            description = og_desc.get("content").strip()

    # title
    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title.get("content").strip()
    if not title and soup.title:
        title = soup.title.string.strip()

    # post-body
    post_body = ""
    body_div = soup.find("div", class_="post-body")
    if body_div:
        post_body = body_div.get_text("\n", strip=True)

    # comments（section.comment-body）
    comments = []
    for section in soup.select("section.comment-body"):
        text = section.get_text("\n", strip=True)
        if text:
            comments.append(text)

    return {
        "title": title,
        "description": description,
        "post_body": post_body,
        "post_comments": comments,
    }

def preprocess_all_html_posts() -> None:
    """批量处理RAW_DIR下所有未处理的html文件"""
    raw_files = list(RAW_DIR.glob("*.html"))
    logger.info(f"Found {len(raw_files)} raw html files.")
    processed_count = 0

    for raw_file in raw_files:
        post_id = raw_file.stem  # 文件名不带后缀
        out_file = PROCESSED_DIR / f"{post_id}.json"
        if out_file.exists():
            continue  # 已处理

        logger.info(f"Processing {raw_file.name}...")
        html_content = raw_file.read_text(encoding="utf-8", errors="ignore")

        # 再做一层clean，避免HTML内的非法字符
        html_content = clean_text(html_content)

        post_info = extract_post_info(html_content)

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(post_info, f, ensure_ascii=False, indent=2)

        processed_count += 1
        logger.info(f"Saved processed JSON to {out_file}")

    logger.info(f"Total processed new files: {processed_count}")


if __name__ == "__main__":
    preprocess_all_html_posts()