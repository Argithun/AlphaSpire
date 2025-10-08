import json
import re
import pandas as pd
import yaml
from pathlib import Path

from utils.config_loader import ConfigLoader
from utils.text_dealer import truncate_text

# --- 路径 ---
BASE_DIR = Path(__file__).resolve().parents[1]
PROMPT_FILE = BASE_DIR / "prompts" / "template_generating.yaml"
FIELDS_DIR = BASE_DIR / "data" / "wq_fields"
TEMPLATE_FIELDS_FILE = BASE_DIR / "data" / "wq_template_fields" / "template_fields.json"
OPERATORS_FILE = BASE_DIR / "data" / "wq_template_operators" / "template_operators.csv"


def build_wq_knowledge_prompt():
    """
    读取 YAML 模板，并根据 config 中启用的数据集构建字段、字段类型、操作符信息，
    渲染 inject_wq_knowledge prompt。
    """

    # 读取模板
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_yaml = yaml.safe_load(f)
    template_str = prompt_yaml.get("inject_wq_knowledge", "")
    if not template_str:
        raise ValueError("inject_wq_knowledge not found in template_generating.yaml")

    # 读取配置：启用的数据集
    enabled_datasets = ConfigLoader.get("enabled_field_datasets", [])
    print(f"🔧 Enabled datasets from config: {enabled_datasets}")

    # =========================================================
    # 加载字段文件（仅限配置中启用的）
    # =========================================================
    field_dfs = []
    for file in FIELDS_DIR.glob("*.csv"):
        dataset_name = file.stem
        if enabled_datasets and dataset_name not in enabled_datasets:
            continue
        if file.stat().st_size == 0:
            print(f"⚠️ Skipping empty file: {file.name}")
            continue

        try:
            df = pd.read_csv(file, dtype=str, keep_default_na=False)
            df["__dataset__"] = dataset_name
            field_dfs.append(df)
        except Exception as e:
            print(f"❌ Failed to load {file.name}: {e}")

    if not field_dfs:
        raise ValueError("❌ No valid field CSVs loaded. Check config.enabled_field_datasets.")

    fields_df = pd.concat(field_dfs, ignore_index=True)

    # 构建字段定义信息
    fields_info = []
    for _, row in fields_df.iterrows():
        desc = row.get("description", "")
        dtype = row.get("type", "")
        dataset = row.get("__dataset__", "")
        field_str = f"- **{row['id']}** ({dtype}, {dataset}): {desc}"
        fields_info.append(field_str)

    fields_and_definitions = "\n".join(fields_info)

    # =========================================================
    # 加载字段类型映射（来自 template_fields.json）
    # =========================================================
    if not TEMPLATE_FIELDS_FILE.exists():
        raise FileNotFoundError(f"❌ template_fields.json not found at {TEMPLATE_FIELDS_FILE}")

    with open(TEMPLATE_FIELDS_FILE, "r", encoding="utf-8") as f:
        template_field_data = json.load(f)

    # template_fields.json 格式: { "field_type_name": [list of field ids], ... }
    # 仅保留属于启用数据集的字段
    filtered_field_types = {}

    for ftype_full, ids in template_field_data.items():
        # 提取 dataset_name，例如从 "</momentum:type:pv1/>" 得到 "pv1"
        match = re.search(r":([\w\-]+)\/>$", ftype_full)
        if not match:
            continue
        dataset_name = match.group(1)
        # 若 dataset_name 在启用列表中，则保留
        if enabled_datasets and dataset_name not in enabled_datasets:
            continue
        filtered_field_types[ftype_full] = ids

    # 渲染 field types
    field_types_str = []
    for ftype, fields in filtered_field_types.items():
        field_types_str.append(f"- **{ftype}**: {', '.join(fields)}")
    field_types = "\n".join(field_types_str)

    # =========================================================
    # 加载操作符文件
    # =========================================================
    ops_df = pd.read_csv(OPERATORS_FILE)
    ops_info = []
    op_types_map = {}

    for _, row in ops_df.iterrows():
        op_str = f"- **{row['name']}**: {row['definition']} — {row['description']}"
        ops_info.append(op_str)
        op_types_map.setdefault(row['type'], []).append(row['name'])

    operators_and_definitions = "\n".join(ops_info)

    op_types_str = []
    for otype, ops in op_types_map.items():
        op_types_str.append(f"- **</{otype}/>**: {', '.join(ops)}")
    operator_types = "\n".join(op_types_str)

    # =========================================================
    # 渲染模板
    # =========================================================
    prompt_filled = (
        template_str
        .replace("{{ fields_and_definitions }}", fields_and_definitions)
        .replace("{{ operators_and_definitions }}", operators_and_definitions)
        .replace("{{ field_types }}", field_types)
        .replace("{{ operator_types }}", operator_types)
    )

    print("✅ WQ knowledge prompt built successfully.")
    return prompt_filled


def build_check_if_blog_helpful(blog_json_path: str):
    """
    从yaml读取check_if_blog_helpful模板并用blog_json渲染
    """
    # 1. 读取yaml
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_yaml = yaml.safe_load(f)

    template_str = prompt_yaml.get("check_if_blog_helpful", "")
    if not template_str:
        raise ValueError("check_if_blog_helpful not found in template_generating.yaml")

    # 2. 读取json
    with open(blog_json_path, "r", encoding="utf-8") as f:
        blog_data = json.load(f)

    # 3. 拼接blog_post文本
    # 可以按需求拼接（title+description+post_body+comments）
    post_text = f"Title: {blog_data.get('title','')}\n\nDescription: {blog_data.get('description','')}\n\nPost Body: {blog_data.get('post_body','')}\n\nComments:\n"
    if blog_data.get("post_comments"):
        for i, c in enumerate(blog_data["post_comments"], 1):
            post_text += f"[{i}] {c}\n"

    # 4. 替换模板
    prompt_filled = template_str.replace("{{ blog_post }}", truncate_text(post_text))

    return prompt_filled


def build_blog_to_hypothesis(blog_json_path: str):
    """
    从yaml读取blog_to_hypothesis模板并用blog_json渲染
    """
    # 1. 读取yaml
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_yaml = yaml.safe_load(f)

    template_str = prompt_yaml.get("blog_to_hypothesis", "")
    if not template_str:
        raise ValueError("blog_to_hypothesis not found in template_generating.yaml")

    # 2. 读取json
    with open(blog_json_path, "r", encoding="utf-8") as f:
        blog_data = json.load(f)

    # 3. 拼接blog_post文本
    # 可以按需求拼接（title+description+post_body+comments）
    post_text = f"Title: {blog_data.get('title','')}\n\nDescription: {blog_data.get('description','')}\n\nPost Body: {blog_data.get('post_body','')}\n\nComments:\n"
    if blog_data.get("post_comments"):
        for i, c in enumerate(blog_data["post_comments"], 1):
            post_text += f"[{i}] {c}\n"

    # 4. 替换模板
    prompt_filled = template_str.replace("{{ blog_post }}", truncate_text(post_text))

    return prompt_filled


def build_hypothesis_to_template(hypotheses_json_path: str):
    """
    从yaml读取hypothesis_to_template模板并用hypotheses_json渲染
    - 优先使用 template_fields.json（映射 field types -> [field ids]）
    - 根据 config 中的 enabled_datasets 过滤 field types （从 </name:type:dataset/> 中解析 dataset）
    - 为防止 token 爆炸，展示每个类型的前 N 个示例并标注总数
    """
    import re
    from utils.config_loader import ConfigLoader

    # 1. 读取yaml
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_yaml = yaml.safe_load(f)

    template_str = prompt_yaml.get("hypothesis_to_template", "")
    if not template_str:
        raise ValueError("hypothesis_to_template not found in template_generating.yaml")

    # 2. 读取hypotheses json
    with open(hypotheses_json_path, "r", encoding="utf-8") as f:
        hypotheses_data = json.load(f)

    hypotheses_str = json.dumps(hypotheses_data, indent=2, ensure_ascii=False)

    # --- 读取并过滤 field types (优先 template_fields.json) ---
    field_types_map = {}

    # 获取用户在 config 中启用的数据集（可为逗号分隔字符串或 list）
    enabled = ConfigLoader.get("enabled_field_datasets")
    enabled_datasets = None
    if enabled:
        if isinstance(enabled, str):
            enabled_datasets = [s.strip() for s in enabled.split(",") if s.strip()]
        elif isinstance(enabled, (list, tuple)):
            enabled_datasets = list(enabled)
        else:
            enabled_datasets = None

    if TEMPLATE_FIELDS_FILE.exists():
        with open(TEMPLATE_FIELDS_FILE, "r", encoding="utf-8") as f:
            template_field_data = json.load(f)

        # 解析 key 格式 </name:type:dataset/> 提取 dataset 并按 enabled_datasets 过滤
        for ftype_full, ids in template_field_data.items():
            # 提取最后一个冒号之后直到 '/>' 之间的 dataset 名称
            m = re.search(r":([^/>]+)\/>$", ftype_full)
            dataset_name = m.group(1) if m else None

            # 如果用户指定了 enabled_datasets，则只保留匹配的 dataset
            if enabled_datasets and dataset_name and dataset_name not in enabled_datasets:
                continue

            # 保持原始 id 列表（list），后面渲染时会截断显示
            field_types_map[ftype_full] = list(ids)
    else:
        raise FileNotFoundError("❌ template_fields.json not found.")

    # --- 读取操作符类型映射（保持原样，但做截断显示） ---
    ops_df = pd.read_csv(OPERATORS_FILE, dtype=str, keep_default_na=False)
    op_types_map = {}
    for _, row in ops_df.iterrows():
        typ = row.get("type", "Other")
        name = row.get("name")
        if name:
            op_types_map.setdefault(typ, []).append(name)

    # --- 构建可读字符串（为 prompt ）: 对每个类型只显示前 N 个示例以节约 token ---
    MAX_EXAMPLES_PER_TYPE = 10000  # 每个类型在 prompt 中展示的最大示例数（字段或操作符）
    field_types_str_lines = []
    for ftype, ids in field_types_map.items():
        total = len(ids)
        display_ids = ids[:MAX_EXAMPLES_PER_TYPE]
        suffix = "" if total <= MAX_EXAMPLES_PER_TYPE else f", ... (+{total - MAX_EXAMPLES_PER_TYPE} more)"
        field_types_str_lines.append(f"- **{ftype}** ({total} fields): {', '.join(display_ids)}{suffix}")
    field_types = "\n".join(field_types_str_lines)

    op_types_str_lines = []
    for otype, ops in op_types_map.items():
        total = len(ops)
        display_ops = ops[:MAX_EXAMPLES_PER_TYPE]
        suffix = "" if total <= MAX_EXAMPLES_PER_TYPE else f", ... (+{total - MAX_EXAMPLES_PER_TYPE} more)"
        op_types_str_lines.append(f"- **</{otype}/>** ({total} ops): {', '.join(display_ops)}{suffix}")
    operator_types = "\n".join(op_types_str_lines)

    # 4. 替换模板
    prompt_filled = (
        template_str
        .replace("{{ hypotheses }}", hypotheses_str)
        .replace("{{ field_types }}", field_types)
        .replace("{{ operator_types }}", operator_types)
    )

    return prompt_filled




if __name__ == "__main__":
    print(build_wq_knowledge_prompt())
