import json

import pandas as pd
import yaml
from pathlib import Path

from utils.text_dealer import truncate_text

# --- 路径 ---
BASE_DIR = Path(__file__).resolve().parents[1]
PROMPT_FILE = BASE_DIR / "prompts" / "template_generating.yaml"
FIELDS_FILE = BASE_DIR / "data" / "wq_template_fields" / "template_fields.csv"
OPERATORS_FILE = BASE_DIR / "data" / "wq_template_operators" / "template_operators.csv"


def build_wq_knowledge_prompt():
    """
    读取yaml模板，并用字段/操作符信息渲染inject_wq_knowledge prompt
    """
    # 1. 读取模板yaml
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_yaml = yaml.safe_load(f)

    template_str = prompt_yaml.get("inject_wq_knowledge", "")
    if not template_str:
        raise ValueError("inject_wq_knowledge not found in template_generating.yaml")

    # 2. 读取字段文件
    fields_df = pd.read_csv(FIELDS_FILE)
    fields_info = []
    field_types_map = {}

    for _, row in fields_df.iterrows():
        field_str = f"- **{row['id']}** ({row['type']}): {row['description']}"
        fields_info.append(field_str)
        field_types_map.setdefault(row['final_category'], []).append(row['id'])

    fields_and_definitions = "\n".join(fields_info)

    # 3. 读取操作符文件
    ops_df = pd.read_csv(OPERATORS_FILE)
    ops_info = []
    op_types_map = {}

    for _, row in ops_df.iterrows():
        op_str = f"- **{row['name']}**: {row['definition']} ——— {row['description']}"
        ops_info.append(op_str)
        op_types_map.setdefault(row['type'], []).append(row['name'])

    operators_and_definitions = "\n".join(ops_info)

    # 4. Field Types
    field_types_str = []
    for ftype, fields in field_types_map.items():
        field_types_str.append(f"- **</{ftype}/>**: {', '.join(fields)}")
    field_types = "\n".join(field_types_str)

    # 5. Operator Types
    op_types_str = []
    for otype, ops in op_types_map.items():
        op_types_str.append(f"- **</{otype}/>**: {', '.join(ops)}")
    operator_types = "\n".join(op_types_str)

    # 6. 渲染模板
    prompt_filled = (
        template_str
        .replace("{{ fields_and_definitions }}", fields_and_definitions)
        .replace("{{ operators_and_definitions }}", operators_and_definitions)
        .replace("{{ field_types }}", field_types)
        .replace("{{ operator_types }}", operator_types)
    )

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
    """
    # 1. 读取yaml
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_yaml = yaml.safe_load(f)

    template_str = prompt_yaml.get("hypothesis_to_template", "")
    if not template_str:
        raise ValueError("hypothesis_to_template not found in template_generating.yaml")

    # 2. 读取json
    with open(hypotheses_json_path, "r", encoding="utf-8") as f:
        hypotheses_data = json.load(f)

    # 3. 把json转成字符串
    #   - 保持原始缩进，使LLM直接看到标准JSON
    hypotheses_str = json.dumps(hypotheses_data, indent=2, ensure_ascii=False)

    # 读取字段文件
    fields_df = pd.read_csv(FIELDS_FILE)
    field_types_map = {}

    for _, row in fields_df.iterrows():
        field_types_map.setdefault(row['final_category'], []).append(row['id'])

    # 读取操作符文件
    ops_df = pd.read_csv(OPERATORS_FILE)
    op_types_map = {}

    for _, row in ops_df.iterrows():
        op_types_map.setdefault(row['type'], []).append(row['name'])

    # Field Types
    field_types_str = []
    for ftype, fields in field_types_map.items():
        field_types_str.append(f"- **</{ftype}/>**: {', '.join(fields)}")
    field_types = "\n".join(field_types_str)

    # Operator Types
    op_types_str = []
    for otype, ops in op_types_map.items():
        op_types_str.append(f"- **</{otype}/>**: {', '.join(ops)}")
    operator_types = "\n".join(op_types_str)

    # 4. 替换模板
    prompt_filled = (template_str
                     .replace("{{ hypotheses }}", hypotheses_str)
                     .replace("{{ field_types }}", field_types)
                     .replace("{{ operator_types }}", operator_types)
                     )

    print(prompt_filled)

    return prompt_filled



if __name__ == "__main__":
    print(build_wq_knowledge_prompt())
