from scraper.preprocess_texts import preprocess_all_html_posts
from scraper.scrap_posts_from_wq import scrape_new_posts
from utils.wq_info_loader import OpAndFeature


if __name__ == "__main__":
    # data scraper ------------------------------------
    # scrape_new_posts(limit=10)
    # preprocess_all_html_posts()

    # alpha researcher --------------------------------
    opAndFeature = OpAndFeature()
    opAndFeature.get_operators()
    opAndFeature.get_data_fields()