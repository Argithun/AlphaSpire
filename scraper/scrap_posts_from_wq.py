"""
scrap_posts_from_wq.py — Playwright版
打开浏览器，手动登录后抓取 WorldQuant Consultant 帖子，支持多页。
"""

import time
import datetime
import csv
from pathlib import Path
from bs4 import BeautifulSoup
from loguru import logger
from playwright.sync_api import sync_playwright

from utils.config_loader import ConfigLoader

# --- 目录 ---
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "wq_posts" / "raw_posts"
RAW_DIR.mkdir(parents=True, exist_ok=True)
INDEX_FILE = RAW_DIR / "index.csv"
COOKIES_FILE = RAW_DIR / "cookies.json"


def _load_existing_ids():
    """加载已抓取帖子ID"""
    if not INDEX_FILE.exists():
        return set()
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["id"] for row in reader}


def _save_index_row(post_meta: dict):
    """追加写入index.csv"""
    file_exists = INDEX_FILE.exists()
    with open(INDEX_FILE, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "title", "url", "time"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(post_meta)


def _save_raw_html(post_id: str, html_content: str):
    """保存原始HTML文件"""
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = RAW_DIR / f"{now_str}_{post_id}.html"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Saved raw HTML to {file_path}")


def scrape_new_posts(limit: int = 20):
    """
    Playwright抓取WorldQuant Consultant新帖子
    """
    topic_url = ConfigLoader.get("worldquant_consultant_posts_url")
    existing_ids = _load_existing_ids()
    new_posts_meta = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        logger.info(f"Navigating to topic page: {topic_url}")
        page.goto(topic_url)

        logger.info("请在浏览器中完成登录，如需。完成后在终端按回车继续...")
        input("已登录并看到帖子列表后按回车...")

        fetched_count = 0
        has_next = True
        while has_next and fetched_count < limit:
            # 等主文档加载
            try:
                page.wait_for_load_state("load", timeout=60000)
            except:
                logger.warning("load_state timeout, continue anyway")
            time.sleep(5)

            html = page.content()
            soup = BeautifulSoup(html, "html.parser")

            # 抽取帖子链接
            post_links = soup.select("a[href*='/community/posts/']")
            logger.info(f"Found {len(post_links)} post links on this page.")

            for link in post_links:
                if fetched_count >= limit:
                    break
                post_url = link.get("href")
                if not post_url:
                    continue
                full_url = (
                    post_url
                    if post_url.startswith("http")
                    else "https://support.worldquantbrain.com" + post_url
                )

                import re
                m = re.search(r"/posts/(\d+)", full_url)
                if not m:
                    continue
                post_id = m.group(1)

                if post_id in existing_ids:
                    context.storage_state(path=str(COOKIES_FILE))
                    logger.info(f"Saved cookies to {COOKIES_FILE}")
                    browser.close()
                    logger.info(f"Total new posts scraped: {len(new_posts_meta)}")
                    return new_posts_meta


                # 抓帖子详情页
                page.goto(full_url)
                try:
                    page.wait_for_load_state("load", timeout=60000)
                except:
                    logger.warning("Timeout loading post page, continue anyway")
                time.sleep(3)
                html_content = page.content()
                _save_raw_html(post_id, html_content)

                title = link.get_text(strip=True)
                post_meta = {
                    "id": post_id,
                    "title": title,
                    "url": full_url,
                    "time": "",
                }
                _save_index_row(post_meta)
                new_posts_meta.append(post_meta)
                fetched_count += 1
                logger.info(f"New post scraped: {post_meta}")

                # 回到列表页
                page.goto(topic_url)
                try:
                    page.wait_for_load_state("load", timeout=60000)
                except:
                    logger.warning("Timeout loading topic page, continue anyway")
                time.sleep(5)

            # 翻页：找“Next”按钮
            next_button = page.locator("a:has-text('Next')")
            if next_button.count() > 0 and fetched_count < limit:
                logger.info("Clicking Next button...")
                next_button.first.click()
                try:
                    page.wait_for_load_state("load", timeout=60000)
                except:
                    logger.warning("Timeout after clicking Next, continue anyway")
                time.sleep(5)
            else:
                has_next = False

        # 保存cookies
        context.storage_state(path=str(COOKIES_FILE))
        logger.info(f"Saved cookies to {COOKIES_FILE}")

        browser.close()

    logger.info(f"Total new posts scraped: {len(new_posts_meta)}")
    return new_posts_meta


if __name__ == "__main__":
    new_posts = scrape_new_posts()
    print(f"Scraped {len(new_posts)} new posts.")
