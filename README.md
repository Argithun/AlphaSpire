# AlphaSpire

AI agent-driven fully automated Alpha mining.

-----

## ⚠️ Attention

Please note that this tool only performs backtest simulations on the platform (Simulate) and does not execute any submission actions (Submit). **Excessive simulations without valid submissions may result in your account being locked.** Please make sure to limit the number of backtests when using this tool.

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
1. Six specialist agents independently extract hypotheses from each helpful post and save them to ./data/hypothesis_db_v2.
2. A LangGraph committee proposes, discusses, revises and votes on templates; the winner is saved to ./data/template_db_v2.
3. Multiple loops generate alphas based on template, field type, and operator type and save them to ./data/alpha_db_v2/all_alphas

### Multi-agent research

The research stage uses LangChain runnable agents orchestrated by a LangGraph
state graph, replacing the previous single-agent LLMChain and conversation memory.
Specialists and all stage prompts are defined in `prompts/multi_agent_research.yaml`:

| Specialist | Research bias |
| --- | --- |
| Equity fundamental | Valuation, accounting quality, cash flows, earnings revisions |
| Options and volatility | Convexity, volatility premia, hedging and tail risk |
| Macro and cross-asset | Cycles, liquidity, sector rotation, transmission effects |
| Statistical arbitrage | Mean reversion, momentum, relative value, normalization |
| Behavioral and event-driven | News, sentiment, earnings events, information diffusion |
| Risk and microstructure | Liquidity, turnover, costs, crowding, downside risk |

At least five uniquely identified specialists are required. These are different
prompt-conditioned research perspectives, not independent trained models. Every
perspective must produce a signal for the existing WQ **EQUITY** simulation setup;
an options specialist does not execute options or invent unavailable derivatives data.

```text
Independent hypotheses (2-3 per agent)
  -> One candidate per agent, informed by all hypotheses and the original post
  -> Peer discussion of every other candidate
  -> Simultaneous revisions using the same pre-revision snapshot
  -> Repeat discussion/revision for the configured number of rounds
  -> Independent anonymous peer scoring and preference ballots
  -> Deterministic ranking and selection
```

Candidate IDs and peer presentation order are shuffled independently of specialist order, and author mappings
are not supplied during voting. This reduces explicit author bias, but cannot
guarantee that an LLM never infers authorship from content. Agents cannot score
their own candidates or see other final ballots. Every candidate must receive
exactly one evaluation from each of the other agents. Duplicate, missing,
out-of-range, nonfinite or self-evaluations are rejected and retried.

Each evaluation scores evidence (25%), economic logic (25%), feasibility (25%),
robustness (15%) and novelty (10%), each from 0 to 10. Candidates rank by average
weighted peer score, then first-choice votes, then lower score standard deviation,
then candidate ID. Each first-choice vote must agree with that voter's maximum
weighted score. Scores are **qualitative research judgments**, not Sharpe,
Fitness or proof of out-of-sample profitability. The existing evaluator remains
responsible for actual backtests.

Configure the committee in `config.yaml`:

```yaml
research:
  discussion_rounds: 1  # 1-3 rounds, each with discussion and revision
  max_workers: 3       # Concurrent requests within each stage (1-6)
  max_retries: 2       # Additional attempts per agent and stage (0-5)
  request_timeout: 120
  max_tokens: 6000     # Maximum output tokens per request
  temperature: 0.2
```

With six agents and one discussion round, a post requires **30 LLM calls** before
retries: six each for hypotheses, proposals, discussion, revision and voting.
Each additional discussion round adds 12 calls. Every stage completes before the
next begins. Invalid JSON, invalid templates and incomplete ballots are retried
with validation feedback; exhausted retries abort that post rather than silently
shrinking the committee. Request-level retries are managed by the workflow.

The winner keeps the existing `SelectedHypothesis`, `TemplateExpression`,
`Description` and `ExpectedBehavior` fields, so alpha expansion and backtesting
consume it unchanged. `ResearchMetadata` adds ranking and the audit-file location.
The audit in `data/research_db_v2/<post>_research.json` records the original post,
specialist profiles, independent hypotheses, candidates before and after each
discussion, all reviews, ballots, author mapping, settings and final ranking.
Files are replaced atomically, and the winning template is written last.

Templates receive local checks for known/nonempty placeholders, 1-4 placeholder
occurrences, known field/operator identifiers, balanced syntax and no scientific
notation. These checks are not a full WQ parser or datatype/signature verifier;
peer review prompts explicitly address those issues, and WQ simulation is still
the final execution check. Repeated placeholders expand independently, as in the
existing alpha generator. The committee receives the full supplied post rather
than the legacy 5,000-character truncation; large posts/field catalogs may require
a model with a larger context window.

Existing template files are skipped by default, including older single-agent
outputs. Use `python main_researcher.py --force` to regenerate all helpful posts
with the committee, or regenerate a particular post:

```python
from researcher.generate_template import from_post_to_template
from researcher.generate_alpha import generate_alphas_from_template

template_file = from_post_to_template("data/wq_posts/helpful_posts/post.json", force=True)
generate_alphas_from_template(template_file)
```

Regeneration does not remove previous backtest CSVs; manage those separately if
you want fresh measurements. To upgrade an existing environment before running
`python main_researcher.py`:

```bash
pip install --upgrade 'langchain>=1,<2' 'langchain-openai>=1,<2' 'langgraph>=1,<2' 'openai>=2,<3'
```

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
