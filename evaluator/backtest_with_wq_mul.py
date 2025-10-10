# backtest_with_wq.py
import json
import csv
import logging
from pathlib import Path
from time import sleep
import requests
from requests.auth import HTTPBasicAuth
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
            pending[sim_id] = {"alpha": alpha_expr, "progress_url": sim_url}

            print(f"📩 提交成功: {i}/{len(alphas)} -> {alpha_expr[:50]}...")

            # 控制批量大小
            if len(pending) >= batch_size:
                monitor_pending(sess, pending, writer)
        except Exception as e:
            logging.error(f"提交 {alpha_expr} 出错: {e}")
            retry_queue.append(alpha_expr)

    # 处理剩余的
    if pending:
        monitor_pending(sess, pending, writer)

    csv_file.close()
    print(f"🎯 回测完成，结果已保存 {out_csv}")
    return str(out_csv)


def monitor_pending(sess, pending, writer):
    """监控 pending 队列直到全部完成"""
    while pending:
        finished_ids = []
        for sim_id, info in list(pending.items()):
            try:
                status_resp = sess.get(info["progress_url"])
                if status_resp.status_code == 429:
                    continue

                status_json = status_resp.json()
                status = status_json.get("status")

                if status == "COMPLETE" or status == "WARNING":
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
                    writer.writerow({
                        "alpha": info["alpha"],
                        "sharpe": None,
                        "turnover": None,
                        "fitness": None,
                        "returns": None,
                        "drawdown": None,
                        "margin": f"FAILED:{status}"
                    })
                    print(f"❌ Simulation failed: {info['alpha']}...")
                    finished_ids.append(sim_id)
                else:
                    print(f"⏳ {info['alpha']} simulation status: {status}")

            except Exception as e:
                logging.error(f"检查 {sim_id} 出错: {e}")

        for fid in finished_ids:
            pending.pop(fid, None)

        sleep(5)


if __name__ == "__main__":
    test_file = BASE_DIR / "data" / "alpha_db" / "all_alphas" / "your_template_alphas.json"
    run_backtest_mul_by_wq_api(test_file)