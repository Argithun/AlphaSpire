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

# ====== 登录并保持 Session ======
def sign_in():
    """登录 WQ Brain 并返回 session"""
    username = ConfigLoader.get('worldquant_account')
    password = ConfigLoader.get('worldquant_password')

    sess = requests.Session()
    sess.auth = HTTPBasicAuth(username, password)
    response = sess.post(ConfigLoader.get('worldquant_api_auth'))
    print(f"Login status: {response.status_code}")
    return sess


def run_backtest_by_wq_api(alphas_json_file):
    """回测指定 alphas json 文件"""
    sess = sign_in()

    # === 1. 读 alpha JSON ===
    with open(alphas_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🔬 Start backtest for {alphas_json_file}")
    # 支持两种结构
    alphas = []
    if "GeneratedAlphas" in data:
        for item in data["GeneratedAlphas"]:
            alphas.append(item["alpha"])
    elif isinstance(data, list):
        for item in data:
            alphas.append(item["alpha"])
    else:
        print("❌ 不识别的 alpha JSON 格式")
        return None

    template_name = Path(alphas_json_file).stem
    out_csv = BACKTEST_DIR / f"{template_name}_backtest.csv"

    # === 2. 读取已存在CSV，跳过已回测 ===
    finished_alphas = set()
    if out_csv.exists():
        with open(out_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                finished_alphas.add(row["alpha"])
        print(f"⚠️ 已有 {len(finished_alphas)} 条回测结果，将跳过这些 alpha")

    # === 3. CSV准备写入 ===
    fieldnames = ["alpha", "sharpe", "turnover", "fitness", "returns", "drawdown", "margin"]
    csv_file = open(out_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if csv_file.tell() == 0:  # 空文件时写表头
        writer.writeheader()

    # === 4. 循环回测 ===
    alpha_fail_attempt_tolerance = 15
    for index, alpha_expr in enumerate(alphas, start=1):
        if alpha_expr in finished_alphas:
            print(f"✅ 跳过已回测 alpha: {alpha_expr[:40]}...")
            continue

        # 组装模拟参数
        alpha_payload = {
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
            "regular": alpha_expr
        }

        print(f"[{index}/{len(alphas)}] 回测 alpha: {alpha_expr[:60]}...")
        keep_trying = True
        failure_count = 0

        # === 4.1 提交 Simulation ===
        while keep_trying:
            try:
                sim_resp = sess.post(
                    'https://api.worldquantbrain.com/simulations',
                    json=alpha_payload
                )
                if sim_resp.status_code not in (200, 201):
                    raise RuntimeError(f"Simulation submit failed {sim_resp.status_code}: {sim_resp.text}")

                sim_progress_url = sim_resp.headers.get('Location')
                if not sim_progress_url:
                    raise RuntimeError("❌ No Location header in response")

                print(f"🔎 Alpha simulation location: {sim_progress_url}")
                keep_trying = False
            except Exception as e:
                failure_count += 1
                print(f"⚠️ No Location, sleep 15 and retry: {e}")
                logging.error(f"No Location, sleep 15 and retry: {e}")
                sleep(15)
                if failure_count >= alpha_fail_attempt_tolerance:
                    sess = sign_in()  # 重新登录
                    failure_count = 0
                    logging.error(f"❌ Too many failures,跳过当前 alpha {alpha_expr}")
                    break

        # === 4.2 轮询 Simulation 结果 ===
        if not sim_progress_url:
            continue
        # 等待完成
        finished = False
        for _ in range(240):  # 最多轮询 240 次 * 15s = 60 分钟
            status_resp = sess.get(sim_progress_url)
            status_json = status_resp.json()
            status = status_json.get("status")
            if status == "COMPLETE":
                alpha_id = status_json.get("alpha")
                finished = True
                break
            elif status == "ERROR":
                print(f"❌ Simulation failed for {alpha_expr}")
                finished = False
                break
            else:
                print(f"⏳ Status: {status}, sleep 10s")
                sleep(10)
        if not finished:
            continue

        # === 4.3 获取 Alpha 指标 ===
        # alpha_resp = sess.get(f'https://api.worldquantbrain.com/alphas/{alpha_id}')
        for attempt in range(20):
            alpha_resp = sess.get(f'https://api.worldquantbrain.com/alphas/{alpha_id}')
            if alpha_resp.status_code == 200:
                alpha_data = alpha_resp.json()
                break
            else:
                print(f"⏳ Alpha {alpha_id} not ready yet, status={alpha_resp.status_code}, retry {attempt + 1}")
                sleep(5)
        else:
            print(f"❌ Failed to fetch alpha result after retries for alphaId={alpha_id}")
            continue  # 或 raise

        is_data = alpha_data.get("is", {})
        result_row = {
            "alpha": alpha_expr,
            "sharpe": is_data.get("sharpe"),
            "turnover": is_data.get("turnover"),
            "fitness": is_data.get("fitness"),
            "returns": is_data.get("returns"),
            "drawdown": is_data.get("drawdown"),
            "margin": is_data.get("margin"),
        }
        writer.writerow(result_row)
        csv_file.flush()
        print(f"✅ 已写入回测结果: sharpe={result_row['sharpe']}, fitness={result_row['fitness']}")


    csv_file.close()
    print(f"🎯 所有回测完成，结果已保存到 {out_csv}")
    return str(out_csv)


if __name__ == "__main__":
    test_file = BASE_DIR / "data" / "alpha_db" / "all_alphas" / "your_template_alphas.json"
    run_backtest_by_wq_api(test_file)