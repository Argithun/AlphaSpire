from researcher.construct_prompts import build_wq_knowledge_prompt, build_check_if_blog_helpful, \
    build_blog_to_hypothesis
from researcher.generate_alpha import generate_alphas_from_template
from researcher.generate_template import from_post_to_template
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

    template_file = from_post_to_template()
    alphas_file = generate_alphas_from_template(template_file)
    