import json
import re

def extract_json(text: str):
    """
    尝试从大模型返回的 text 中提取并解析 JSON。
    - 自动去掉 Markdown 代码块、注释、解释文字
    - 尝试匹配第一个 {...} 或 [...] 的完整 JSON
    返回 Python 对象
    """
    if not text:
        raise ValueError("❌ Empty text, cannot parse JSON")

    # 1. 去掉Markdown代码块 ```json ... ```
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    # 2. 在字符串中找到第一个 { 或 [
    start = min(
        (cleaned.find("{") if "{" in cleaned else float("inf")),
        (cleaned.find("[") if "[" in cleaned else float("inf")),
    )
    if start == float("inf"):
        raise ValueError(f"❌ No JSON start symbol found in: {text[:200]}")

    # 3. 截取可能的JSON部分
    candidate = cleaned[start:]

    # 4. 尝试从后往前找到匹配的 } 或 ]
    end_brace = candidate.rfind("}")
    end_bracket = candidate.rfind("]")
    end = max(end_brace, end_bracket)
    if end == -1:
        raise ValueError(f"❌ No JSON end symbol found in: {text[:200]}")

    candidate = candidate[:end + 1]

    # 5. 尝试解析
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ JSON decode error: {e}\nExtracted candidate:\n{candidate[:500]}")
