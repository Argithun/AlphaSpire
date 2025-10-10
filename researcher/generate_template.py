import random
import json
from pathlib import Path

from researcher.construct_prompts import build_wq_knowledge_prompt, build_blog_to_hypothesis, \
    build_hypothesis_to_template, build_check_if_blog_helpful
from utils.config_loader import ConfigLoader

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

from utils.json_dealer import extract_json

# --- 路径 ---
BASE_DIR = Path(__file__).resolve().parents[1]
POSTS_DIR = BASE_DIR / "data" / "wq_posts" / "helpful_posts"
HYPOTHESIS_DB = BASE_DIR / "data" / "hypothesis_db_v2"
TEMPLATE_DB = BASE_DIR / "data" / "template_db_v2"
PROMPT_FILE = BASE_DIR / "prompts" / "template_generating.yaml"

HYPOTHESIS_DB.mkdir(parents=True, exist_ok=True)
TEMPLATE_DB.mkdir(parents=True, exist_ok=True)


# === LangChain Agent 初始化 ===
def init_agent(system_prompt):
    """初始化长时运行的Agent并注入System Prompt"""
    base_url = ConfigLoader.get("openai_base_url")
    api_key = ConfigLoader.get("openai_api_key")
    model_name = ConfigLoader.get("openai_model_name")

    llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        temperature=0.2,
    )

    # memory 保存上下文
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # PromptTemplate — system prompt + user placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")  # 后续只传 input
    ])

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )

    return chain


# === 随机选择有用的 Blog Post ===
def select_valid_post(chain):
    post_files = list(POSTS_DIR.glob("*.json"))
    if not post_files:
        raise FileNotFoundError("❌ No blog post found in processed_posts folder")

    # while True:
    #     post_file = random.choice(post_files)
    #     formatted = build_check_if_blog_helpful(post_file)
    #     output = chain.run(input=formatted).strip()
    #
    #     if output == "Y":
    #         print(f"✅ Selected blog post: {post_file}")
    #         return post_file
    #     else:
    #         print(f"⚠️ Skipping blog post: {post_file} (not helpful)")
    return random.choice(post_files)


def check_if_post_helpful(chain, post_file):
    formatted = build_check_if_blog_helpful(post_file)
    output = chain.run(input=formatted).strip()
    if output.upper().startswith("Y"):
        return True
    else:
        return False


# === 生成 Hypotheses ===
def generate_hypotheses(chain, post_file):
    formatted = build_blog_to_hypothesis(post_file)
    output = chain.run(input=formatted).strip()

    try:
        hypotheses = extract_json(output)
    except Exception:
        raise ValueError(f"❌ Hypotheses output not valid JSON: {output}")

    out_file = HYPOTHESIS_DB / f"{Path(post_file).stem}_hypotheses.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(hypotheses, f, indent=2, ensure_ascii=False)

    print(f"✅ Hypotheses saved: {out_file}")
    return out_file


# === 生成 Template ===
def generate_template(chain, hypotheses_file):
    formatted = build_hypothesis_to_template(hypotheses_file)
    output = chain.run(input=formatted).strip()

    try:
        template_json = extract_json(output)
    except Exception:
        print(f"❌ Template output not valid JSON: {output}")
        return None

    out_file = TEMPLATE_DB / f"{Path(hypotheses_file).stem}_template.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(template_json, f, indent=2, ensure_ascii=False)

    print(f"✅ Template saved: {out_file}")
    return out_file


# === 主流程 ===
def from_post_to_template(post_file: str=None):
    system_prompt = build_wq_knowledge_prompt()

    chain = init_agent(system_prompt)

    # Step 1: 选择 blog
    if post_file:
        post_stem = Path(post_file).stem
        existing_template = TEMPLATE_DB / f"{post_stem}_hypotheses_template.json"
        if existing_template.exists():
            print(f"✅ Template already exists for {post_file}, skipping template and alpha generation.")
            return None
        blog_file = post_file

        # if check_if_post_helpful(chain, post_file):
        #     blog_file = post_file
        # else:
        #     print(f"⚠️ Skipping blog post: {post_file} (not helpful)")
        #     return None
    else:
        blog_file = select_valid_post(chain)

    # Step 2: 生成 Hypotheses
    hypotheses_file = generate_hypotheses(chain, blog_file)

    # Step 3: 生成 Template
    template_file = generate_template(chain, hypotheses_file)

    print(f"🎯 Finished: Template generated from {blog_file} successfully.")
    return template_file


if __name__ == "__main__":
    from_post_to_template()
