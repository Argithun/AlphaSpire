# generate_alpha.py
import json
import re
import csv
from itertools import product
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
OPERATORS_FILE = BASE_DIR / "data" / "wq_template_operators" / "template_operators.csv"
FIELDS_FILE = BASE_DIR / "data" / "wq_template_fields" / "template_fields.json"
ALPHA_DB = BASE_DIR / "data" / "alpha_db_v2" / "all_alphas"
ALPHA_DB.mkdir(parents=True, exist_ok=True)

# === 最大单次生成 alpha 数量限制 ===
MAX_ALPHAS = 100000


def load_operator_type_map():
    """读取 template_operators.csv，返回 {type: [name,...]}"""
    operator_map = {}
    with open(OPERATORS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_type = row["type"].strip()
            name = row["name"].strip()
            operator_map.setdefault(op_type, []).append(name)
    return operator_map


def load_field_type_map():
    """读取 template_fields.json，返回 {final_category: [id,...]}"""
    handled_field_map = {}
    with open(FIELDS_FILE, "r", encoding="utf-8") as f:
        field_map = json.load(f)
    for raw_type, ids in field_map.items():
        # 提取 </type_name:TYPE:dataset/> 中的核心部分
        clean_type = re.sub(r"^</|/>$", "", raw_type).strip()
        handled_field_map[clean_type] = ids
    return handled_field_map



def extract_placeholders(expression):
    """提取 </.../> 的占位符"""
    return re.findall(r"</(.*?)/>", expression)


def generate_alphas_from_template(template_path):
    """从alpha_template.json生成所有具体alpha"""
    # === 加载模板 ===
    with open(template_path, "r", encoding="utf-8") as f:
        template_json = json.load(f)

    template_expr = template_json["TemplateExpression"]
    template_name = Path(template_path).stem

    # === 加载映射 ===
    operator_map = load_operator_type_map()
    field_map = load_field_type_map()

    # === 提取占位符 ===
    placeholders = extract_placeholders(template_expr)
    if not placeholders:
        print("❌ 模板中未发现占位符，无法展开；直接使用 template 作为 alpha")
        all_alphas = []
        all_alphas.append({"alpha": template_expr, "fields_or_ops_used": []})
        out_file = ALPHA_DB / f"{template_name}_alphas.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "Template": template_expr,
                "GeneratedAlphas": all_alphas
            }, f, indent=2, ensure_ascii=False)
        print(f"✅ Generated 1 alphas saved to {out_file}")
        return out_file

    # === 为每个占位符构建替代列表 ===
    replacements_list = []
    for ph in placeholders:
        # 判断是操作符还是字段
        if ph in operator_map:
            replacements_list.append(operator_map[ph])
        elif ph in field_map:
            replacements_list.append(field_map[ph])
        else:
            print(f"❌ 未在字段或操作符映射中找到占位符类型: {ph}")
            return None

    # === 笛卡尔积替换 ===
    all_alphas = []
    count = 0
    total_combinations = 1
    for lst in replacements_list:
        total_combinations *= len(lst)
    if total_combinations > MAX_ALPHAS:
        print(f"⚠️ Warning: Total possible alphas {total_combinations} exceeds MAX_ALPHAS={MAX_ALPHAS}. "
              f"Only generating the first {MAX_ALPHAS} combinations.")

    for combo in product(*replacements_list):
        expr = template_expr
        # 依次替换占位符
        for ph, val in zip(placeholders, combo):
            expr = re.sub(rf"</{re.escape(ph)}/>+", val, expr, count=1)
        all_alphas.append({
            "alpha": expr,
            "fields_or_ops_used": combo
        })
        count += 1
        if count >= MAX_ALPHAS:  # 超过限制提前退出
            break

    # === 保存 ===
    out_file = ALPHA_DB / f"{template_name}_alphas.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "Template": template_expr,
            "GeneratedAlphas": all_alphas
        }, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated {len(all_alphas)} alphas saved to {out_file}")
    return out_file


if __name__ == "__main__":
    test_template = BASE_DIR / "data" / "template_db" / "your_alpha_template.json"
    generate_alphas_from_template(test_template)
