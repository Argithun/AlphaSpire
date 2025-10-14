import random
from pathlib import Path

from evaluator.backtest_with_wq import run_backtest_by_wq_api
from evaluator.backtest_with_wq_mul import run_backtest_mul_by_wq_api
from researcher.construct_prompts import build_wq_knowledge_prompt, build_check_if_blog_helpful, \
    build_blog_to_hypothesis
from researcher.generate_alpha import generate_alphas_from_template
from researcher.generate_template import from_post_to_template
from scraper.preprocess_texts import preprocess_all_html_posts
from scraper.scrap_posts_from_wq import scrape_new_posts
from utils.template_field_gener import generate_template_fields_v2
from utils.template_op_gener import generate_template_ops
from utils.wq_info_loader import OpAndFeature


if __name__ == "__main__":
    # data scraper ------------------------------------
    # scrape_new_posts(limit=50)
    # preprocess_all_html_posts()


    # alpha researcher --------------------------------
    # opAndFeature = OpAndFeature()
    # opAndFeature.get_operators()
    # opAndFeature.get_data_fields()
    #
    # generate_template_ops()
    # generate_template_fields_v2()

    # POSTS_DIR = Path("data/wq_posts/helpful_posts")
    # for json_file in POSTS_DIR.glob("*.json"):
    #
    #     template_file = from_post_to_template(str(json_file))
    #     if template_file is None:
    #         continue
    #     alphas_file = generate_alphas_from_template(template_file)


    # alpha evaluator ----------------------------------
    ALPHA_DIR = Path("data/alpha_db/all_alphas")
    json_files = list(ALPHA_DIR.glob("*.json"))
    random.shuffle(json_files)
    for json_file in json_files:
        backtest_result = run_backtest_mul_by_wq_api(json_file)