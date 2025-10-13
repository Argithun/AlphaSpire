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

## Deployment

```bash
conda env create -f environment.yml
```