# Ranking Experiment

Tools for ranking model-generated solutions across very different domains:
* **IMO-style math proofs** pulled from MathArena (via `parse_imo.py`/`main.py`).
* **Creative-writing submissions** scraped from Reedsy contests and Literary Hub.

The `main.py` entry point can ingest existing JSON datasets or scrape fresh creative samples, then run scoring/ranking pipelines (score-based, Elo, pairwise LLM judging, rubric LLM scoring).

---

## 1. Environment

```bash
uv sync                 # installs core dependencies (tqdm, etc.)
source .venv/bin/activate
```

Optional scraping dependencies (only needed for the creative-writing scrapers):

```bash
uv pip install requests beautifulsoup4
```

Install pre-commit hooks if you plan to contribute:

```bash
pre-commit install
```

---

## 2. Datasets

### 2.1 Existing JSON

`imo_2025_solutions.json` is generated with `parse_imo.py` and already wired into `main.py`. Use `--solutions-json` to point at any alternative dataset (e.g. sampled variants, creative exports).

### 2.2 Scraping MathArena (optional refresh)

```bash
python parse_imo.py -o imo_2025_solutions.json
```

### 2.3 Reedsy Contest Scraper

Use the dedicated CLI to fetch contest prompts/stories and emit a ranking-ready JSON:

```bash
python scrape_reedsy.py 316 315 \
  --regular-count 3 \
  --output data/reedsy_dataset.json
```

`--regular-count` controls how many non-awarded entries are included alongside the winner/featured pieces (default: 2). By default the script writes `reedsy_dataset.json` in the project root.

The generated file follows the same structure as the IMO dataset, so you can load it with `main.py --dataset creative_writing --solutions-json data/reedsy_dataset.json`.

> Need Literary Hub stories or custom blends? Import the helpers from `creative_dataset.py` and assemble datasets programmatically (see `scrape_reedsy.py` for a reference).

---

## 3. Ranking Pipelines

The core CLI:

```bash
python main.py \
  --dataset imo_2025 \
  --ranking-scheme score \
  --judge score \
  --score-seed 13 \
  --output run_rankings.json \
  --summary-output summary.json
```

Key flags:

| Flag / Option            | Description |
|--------------------------|-------------|
| `--dataset`              | `imo_2025` or `creative_writing` (adds more if you drop new JSON in the repo and extend `DATASETS`). |
| `--solutions-json PATH`  | Override dataset file path (useful for freshly scraped creative data). |
| `--ranking-scheme`       | `score`, `elo`, `pairwise`, `llm_score`. |
| `--judge`                | `score` (numeric fallback) or `llm` (requires OpenAI key). |
| `--k-factor`             | Elo updates (only for `--ranking-scheme=elo`). |
| `--llm-model` / `--llm-workers` | Configure LLM judge/scorer. Requires `OPENAI_API_KEY` or `.openai-key`. |
| `--output`               | Persist per-problem ranked runs. |
| `--summary-output`       | Persist aggregated summary metrics. |

### 3.1 LLM Usage

For `--judge llm` or `--ranking-scheme llm_score`, install the OpenAI Python client and set credentials:

```bash
uv pip install openai
export OPENAI_API_KEY=sk-...
# or drop the key in .openai-key
```

`main.py` adapts prompts to the dataset:
* Math competition mode uses rigorous verification prompts.
* Creative-writing mode uses editorial evaluation with a 4-part rubric.

### 3.2 Convenience Script

To run both the pairwise (score-judged) and score-based rankings for a creative-writing dataset and capture summaries in `summary.json` and `summary2.json`:

```bash
./run_creative_rankings.sh data/reedsy_dataset.json
```

Omit the argument to fall back to `data/reedsy_dataset.json`. The script assumes the dataset exists (use `scrape_reedsy.py` or your own builder first).

---

## 4. Repo Tips

- Stored datasets (`imo_2025_solutions.json`, `summary*.json`) are examples; feel free to regenerate.
- Ranking outputs keep solution text and metadata (title/author/source) so you can feed results into downstream analysis.
- The project was bootstrapped from the “tufa labs” template; only the instructions above reflect the current workflow.

For anything else—new data sources, alternate ranking metrics, or automation—open an issue or extend the CLI following the existing patterns.
