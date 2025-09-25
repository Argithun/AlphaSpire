from researcher.construct_prompts import build_wq_knowledge_prompt, build_check_if_blog_helpful, \
    build_blog_to_hypothesis
from scraper.preprocess_texts import preprocess_all_html_posts
from scraper.scrap_posts_from_wq import scrape_new_posts
from utils.template_field_gener import generate_template_fields
from utils.template_op_gener import generate_template_ops
from utils.wq_info_loader import OpAndFeature


if __name__ == "__main__":
    # data scraper ------------------------------------
    # scrape_new_posts(limit=10)
    # preprocess_all_html_posts()

    # alpha researcher --------------------------------
    # opAndFeature = OpAndFeature()
    # opAndFeature.get_operators()
    # opAndFeature.get_data_fields()

    # generate_template_ops()
    # generate_template_fields()

    # print(build_wq_knowledge_prompt())
    # print(build_check_if_blog_helpful("./data/wq_posts/processed_posts/20250921_231726_35081418033047.json"))
    print(build_blog_to_hypothesis("./data/wq_posts/processed_posts/20250921_231726_35081418033047.json"))

