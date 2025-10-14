import yaml
from pathlib import Path

# --- 路径 ---
BASE_DIR = Path(__file__).resolve().parents[1]
PROMPT_FILE = BASE_DIR / "prompts" / "template_evaluating.yaml"


def build_fix_fast_expression_prompt(alpha_expression : str, error_mes : str):
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_yaml = yaml.safe_load(f)
    template_str = prompt_yaml.get("fix_fast_expression", "")
    if not template_str:
        raise ValueError("fix_fast_expression not found in template_generating.yaml")

    prompt_filled = (
        template_str
        .replace("{{ fast_expression }}", alpha_expression)
        .replace("{{ error_mes }}", error_mes)
    )
    return prompt_filled