# AlphaSpire

AI agent-driven fully automated Alpha mining.

-----

## Work Pipeline

```scss
                ┌────────────────────────────┐
                │   Data Acquisition Module  │
                │ (Scraper & Local Storage)  │
                └─────────────┬──────────────┘
                              │
                ┌─────────────▼──────────────┐
                │   Preprocessing & Parsing  │
                │ (Text cleaning, Metadata)  │
                └─────────────┬──────────────┘
                              │
                ┌─────────────▼──────────────┐
                │     Researcher (LLM)       │
                │ (LangChain / Prompting)    │
                └─────────────┬──────────────┘
                              │
                ┌─────────────▼──────────────┐
                │  Alpha Template Generating │
                │ (Factor/Signal Extraction) │
                └─────────────┬──────────────┘
                              │
                ┌─────────────▼──────────────┐
                │          Evaluator         │
                │ (Submit to WQ or local sim)│
                └─────────────┬──────────────┘
                              │
                ┌─────────────▼──────────────┐
                │    Results Collecting      │
                │ (Store Scores, Rank Alphas)│
                └─────────────┬──────────────┘
                              │
                ┌─────────────▼──────────────┐
                │Results analyzing &Iteration│
                │                            │
                └────────────────────────────┘
```

## Architecture
### data
The data folder stores posts, operators, fields, operator types, field types, generated alpha templates, generated alphas, and backtesting results information.

### prompts
Stores prompt templates used in workflows.

### utils
1. Get the operator set of Fast Expression and save it to ./data/wq_operators
2. Get the field collection of Fast Expression and save it to ./data/wq_fields
3. Manual rules classify operators and generate operator types required by templates. (save to ./data/wq_template_operators)
4. Clustering and large language models are used to assist in generating the field types used in templates. (save to ./data/wq_template_fields)
5. Various text processing and other miscellaneous.

### scraper
1. Scrape posts from the WorldQuant forum and store them in ./data/wq_posts/raw_posts.
2. Extract the main information from the original html text of the post into a json file and store it in ./data/wq_posts/processed_posts
3. Use a large language model to determine whether the post has the potential to generate alpha, and if so, save it to ./data/wq_posts/helpful_posts

### researcher
1. Use LLM to summarize hypothesis from blog and save it to ./data/hypothesis_db_v2
2. Use LLM to generate template from hypothesis and save it to ./data/template_db_v2
3. Multiple loops generate alphas based on template, field type, and operator type and save them to ./data/alpha_db_v2/all_alphas

### evaluator
Use the WorldQuant backtesting API to evaluate alpha performance and save the results to data/alpha_db_v2/backtest_result


## Deployment

1. Create a conda environment
    ```bash
    conda env create -f environment.yml
    conda activate alphaspire
    ```
2. Fill in the configuration file (config.yaml)
    ```yaml
    # ===============================
    # Global Configuration File
    # ===============================
    
    # --- OpenAI API Settings ---
    openai_base_url: "todo" # e.g. https://api.deepseek.com
    openai_api_key: "todo" # e.g. sk-...
    openai_model_name: "todo" # e.g. deepseek-chat
    reasoner_model_name: "todo" # e.g. deepseek-reasoner
    
    # --- WorldQuant Platform Credentials ---
    worldquant_account: "todo"
    worldquant_password: "todo"
    
    worldquant_login_url: "https://platform.worldquantbrain.com/sign-in"
    worldquant_api_auth: "https://api.worldquantbrain.com/authentication"
    worldquant_consultant_posts_url: "https://support.worldquantbrain.com/hc/en-us/community/topics/18910956638743-顾问专属中文论坛"
    # You can also choose any other WorldQuant Forum URL you have access to.
    
    # --- Dataset from WorldQuant Brain
    enabled_field_datasets: # Select the field database you want to use to build alphas.
      - pv1                 # Database name reference ./data/wq_fields
      - fundamental6
      - analyst4
      - model16
      - news12   
    ```

3. One-click operation
   * Full process operation
       ```bash
        python3 main.py
       ```
   * Only run post crawl
       ```bash
        python3 main_scraper.py
       ```
   * Only run Alpha generation
       ```bash
        python3 main_researcher.py
       ```
   * Only run Alpha backtests
       ```bash
        ./test_script.sh
       ```
     OR
       ```bash
        python3 main_evaluator.py
       ```

## Notice

Regarding WorldQuant backtest parameter settings, currently only manual modification of the parameters in the following code section in evaluator/backtest_with_wq_mul.py is supported. 
These will be moved to a separate config file in future versions.

```
payload = {
    "type": "REGULAR",
    "settings": {
        "instrumentType": "EQUITY",
        "region": "ASI",
        "universe": "MINVOL1M",
        "delay": 1,
        "decay": 6,
        "neutralization": "SUBINDUSTRY",
        "truncation": 0.01,
        "pasteurization": "ON",
        "unitHandling": "VERIFY",
        "nanHandling": "ON",
        "maxTrade": "ON",
        "language": "FASTEXPR",
        "visualization": False,
    },
    "regular": fixed_expr
}
```
