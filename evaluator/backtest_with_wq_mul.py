# backtest_with_wq.py
import json
import csv
import logging
from pathlib import Path
from time import sleep
import requests
from openai import OpenAI
from requests.auth import HTTPBasicAuth

from evaluator.construct_prompts import build_fix_fast_expression_prompt
from utils.config_loader import ConfigLoader

BASE_DIR = Path(__file__).resolve().parents[1]
BACKTEST_DIR = BASE_DIR / "data" / "alpha_db_v2" / "backtest_result"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename='backtest_with_wq.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def sign_in():
    """登录 WQ Brain 并返回 session"""
    username = ConfigLoader.get('worldquant_account')
    password = ConfigLoader.get('worldquant_password')

    sess = requests.Session()
    sess.auth = HTTPBasicAuth(username, password)
    resp = sess.post(ConfigLoader.get('worldquant_api_auth'))
    print(f"Login status: {resp.status_code}")
    return sess


def run_backtest_mul_by_wq_api(alphas_json_file, batch_size=15):
    """批量回测指定 alphas json 文件，采用等待队列方式提升效率"""
    sess = sign_in()

    # === 1. 读 alpha JSON ===
    with open(alphas_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🔬 Start backtest for {alphas_json_file}")
    if "GeneratedAlphas" in data:
        alphas = [item["alpha"] for item in data["GeneratedAlphas"]]
    elif isinstance(data, list):
        alphas = [item["alpha"] for item in data]
    else:
        print("❌ 不识别的 alpha JSON 格式")
        return None

    template_name = Path(alphas_json_file).stem
    out_csv = BACKTEST_DIR / f"{template_name}_backtest.csv"

    # === 2. 已有结果，跳过 ===
    finished_alphas = set()
    if out_csv.exists():
        with open(out_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                finished_alphas.add(row["alpha"])
        print(f"⚠️ 已有 {len(finished_alphas)} 条回测结果，将跳过这些 alpha")

    # === 3. 准备写入 ===
    fieldnames = ["alpha", "sharpe", "turnover", "fitness", "returns", "drawdown", "margin"]
    csv_file = open(out_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if csv_file.tell() == 0:
        writer.writeheader()

    # === 4. 构造 payload 模板 ===
    def make_payload(expr):
        return {
            "type": "REGULAR",
            "settings": {
                "instrumentType": "EQUITY",
                "region": "USA",
                "universe": "TOP3000",
                "delay": 1,
                "decay": 0,
                "neutralization": "SUBINDUSTRY",
                "truncation": 0.01,
                "pasteurization": "ON",
                "unitHandling": "VERIFY",
                "nanHandling": "OFF",
                "language": "FASTEXPR",
                "visualization": False,
            },
            "regular": expr
        }

    # === 5. 提交 & 管理 pending 队列 ===
    pending = {}  # sim_id -> {"alpha": expr, "progress_url": url}
    retry_queue = []

    for i, alpha_expr in enumerate(alphas, 1):
        if alpha_expr in finished_alphas:
            continue

        # 提交 alpha
        try:
            resp = sess.post("https://api.worldquantbrain.com/simulations", json=make_payload(alpha_expr))
            if resp.status_code not in (200, 201):
                if "SIMULATION_LIMIT_EXCEEDED" in resp.text:
                    retry_queue.append(alpha_expr)
                    continue
                print(f"❌ 提交失败: {resp.status_code}, {resp.text}")
                continue

            sim_url = resp.headers.get("Location")
            if not sim_url:
                retry_queue.append(alpha_expr)
                continue

            sim_id = sim_url.split("/")[-1]
            pending[sim_id] = {"alpha": alpha_expr, "progress_url": sim_url, "first_time": True}

            print(f"📩 提交成功: {i}/{len(alphas)} -> {alpha_expr[:50]}...")

            # 控制批量大小
            if len(pending) >= batch_size:
                monitor_pending(sess, pending, writer, alphas_json_file)
        except Exception as e:
            logging.error(f"提交 {alpha_expr} 出错: {e}")
            retry_queue.append(alpha_expr)

    # 处理剩余的
    if pending:
        monitor_pending(sess, pending, writer, alphas_json_file)

    csv_file.close()
    print(f"🎯 回测完成，结果已保存 {out_csv}")
    return str(out_csv)


def monitor_pending(sess, pending, writer, alphas_json_file):
    """监控 pending 队列直到全部完成"""
    client = OpenAI(
        base_url=ConfigLoader.get("openai_base_url"),
        api_key=ConfigLoader.get("openai_api_key"),
    )

    while pending:
        finished_ids = []
        for sim_id, info in list(pending.items()):
            try:
                status_resp = sess.get(info["progress_url"])
                if status_resp.status_code == 429:
                    continue

                status_json = status_resp.json()
                status = status_json.get("status")

                if status in ("COMPLETE", "WARNING"):
                    alpha_id = status_json.get("alpha")
                    if not alpha_id:
                        finished_ids.append(sim_id)
                        continue

                    # 获取结果
                    alpha_data = None
                    for _ in range(10):
                        alpha_resp = sess.get(f"https://api.worldquantbrain.com/alphas/{alpha_id}")
                        if alpha_resp.status_code == 200:
                            alpha_data = alpha_resp.json()
                            break
                        sleep(3)

                    if not alpha_data:
                        finished_ids.append(sim_id)
                        continue

                    is_data = alpha_data.get("is", {})
                    writer.writerow({
                        "alpha": info["alpha"],
                        "sharpe": is_data.get("sharpe"),
                        "turnover": is_data.get("turnover"),
                        "fitness": is_data.get("fitness"),
                        "returns": is_data.get("returns"),
                        "drawdown": is_data.get("drawdown"),
                        "margin": is_data.get("margin"),
                    })
                    finished_ids.append(sim_id)
                    print(f"✅ 完成: {info['alpha']}... fitness={is_data.get('fitness')}")

                elif status == "ERROR":
                    if info["first_time"]: # 失败直接退出，修复带来的收益过低，时间损耗过高 TODO
                        # 二次失败，写入 None
                        writer.writerow({
                            "alpha": info["alpha"],
                            "sharpe": None,
                            "turnover": None,
                            "fitness": None,
                            "returns": None,
                            "drawdown": None,
                            "margin": f"FAILED:{status}"
                        })
                        print(f"❌ 二次失败: {info['alpha'][:60]}...")
                        finished_ids.append(sim_id)
                    else:
                        # === 使用 LLM 修复表达式 ===
                        print(f"❌ 模拟失败: {info['alpha'][:60]}...")
                        fix_exp_prompt = build_fix_fast_expression_prompt(info["alpha"], str(status_json))
                        try:
                            resp = client.chat.completions.create(
                                model=ConfigLoader.get("reasoner_model_name"),
                                messages=[
                                    {"role": "system", "content": "You are an expert in Fast Expression syntax repair."},
                                    {"role": "user", "content": fix_exp_prompt}
                                ],
                                temperature=0.2,
                            )
                            fixed_expr = resp.choices[0].message.content.strip()
                            print(f"🧩 修复后的表达式: {fixed_expr}")

                            # === 替换 alphas_json_file 文件中的旧 alpha
                            try:
                                with open(alphas_json_file, "r", encoding="utf-8") as f:
                                    text = f.read()
                                if info["alpha"] not in text:
                                    print("⚠️ 原始表达式未在文件中找到，跳过替换")
                                else:
                                    new_text = text.replace(info["alpha"], fixed_expr, 1)  # 仅替换第一次出现
                                    with open(alphas_json_file, "w", encoding="utf-8") as f:
                                        f.write(new_text)
                                    print(f"💾 已在 {alphas_json_file} 中替换修复后的表达式")
                            except Exception as e:
                                print(f"❌ 替换 {alphas_json_file} 中表达式失败: {e}")

                            # === 再次提交修复后的表达式 ===
                            payload = {
                                "type": "REGULAR",
                                "settings": {
                                    "instrumentType": "EQUITY",
                                    "region": "USA",
                                    "universe": "TOP3000",
                                    "delay": 1,
                                    "decay": 0,
                                    "neutralization": "SUBINDUSTRY",
                                    "truncation": 0.01,
                                    "pasteurization": "ON",
                                    "unitHandling": "VERIFY",
                                    "nanHandling": "OFF",
                                    "language": "FASTEXPR",
                                    "visualization": False,
                                },
                                "regular": fixed_expr
                            }

                            new_resp = sess.post("https://api.worldquantbrain.com/simulations", json=payload)
                            if new_resp.status_code not in (200, 201):
                                print(f"⚠️ 修复后提交失败 {new_resp.status_code}: {new_resp.text}")
                                writer.writerow({
                                    "alpha": info["alpha"],
                                    "sharpe": None,
                                    "turnover": None,
                                    "fitness": None,
                                    "returns": None,
                                    "drawdown": None,
                                    "margin": "FIX_FAIL_SUBMIT"
                                })
                                finished_ids.append(sim_id)
                                continue

                            new_url = new_resp.headers.get("Location")
                            if not new_url:
                                print("⚠️ 修复后提交未返回Location，跳过")
                                finished_ids.append(sim_id)
                                continue

                            # 替换原 pending 任务为新任务
                            new_id = new_url.split("/")[-1]
                            pending[new_id] = {
                                "alpha": fixed_expr,
                                "progress_url": new_url,
                                "first_time": False  # 标记为已修复
                            }
                            finished_ids.append(sim_id)
                            print(f"🔁 已重新提交修复后的表达式 {new_id}")

                        except Exception as e:
                            logging.error(f"修复表达式失败: {e}")
                            writer.writerow({
                                "alpha": info["alpha"],
                                "sharpe": None,
                                "turnover": None,
                                "fitness": None,
                                "returns": None,
                                "drawdown": None,
                                "margin": "FIX_FAIL_LLM"
                            })
                            finished_ids.append(sim_id)

                else:
                    print(f"⏳ {info['alpha'][:40]}... simulation status: {status}")

            except Exception as e:
                logging.error(f"检查 {sim_id} 出错: {e}")

        for fid in finished_ids:
            pending.pop(fid, None)

        sleep(5)


if __name__ == "__main__":
    test_file = BASE_DIR / "data" / "alpha_db" / "all_alphas" / "your_template_alphas.json"
    run_backtest_mul_by_wq_api(test_file)