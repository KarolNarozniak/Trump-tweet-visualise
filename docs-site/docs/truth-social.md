# Truth Social Semantic Graph

The Truth Social pipeline is a separate Layer 2 path. It does not replace the original Twitter mention graph.

## Build Command

```bash
python -m trump_graph build-truth \
  --input "truthsocial.posts[Trump-FROM-10-8-25].txt" \
  --out "data/processed_truth"
```

For smoke tests without Hugging Face model downloads:

```bash
python -m trump_graph build-truth \
  --input "tests/fixtures/sample_truth_posts.csv" \
  --out "data/processed_truth" \
  --semantic-backend deterministic \
  --min-node-count 1
```

## Local Models

Default Hugging Face models:

- topics: `facebook/bart-large-mnli`
- entities: `dslim/bert-base-NER`
- sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- embeddings: `sentence-transformers/all-MiniLM-L6-v2`

Install optional ML dependencies:

```powershell
.\scripts\setup_windows.ps1 -InstallMl
```

```bash
./scripts/setup_ubuntu.sh --install-ml
```

For CUDA-specific PyTorch wheels, use the install command recommended by the PyTorch selector for your driver/CUDA version, then rerun `pip install -r requirements-ml.txt` for the remaining packages.

## Temporal Semantics

Truth post IDs are decoded as Mastodon-style snowflake timestamps:

```text
created_at_ms = post_id >> 16
```

Posts are bucketed by ISO week. ReTruths are kept and flagged with `is_retruth`, so the app can include or exclude them.

## Graph Nodes

Semantic nodes are typed:

- `topic::<label>`
- `per::<name>`
- `org::<name>`
- `loc::<name>`
- `hashtag::<tag>`
- `mention::<handle>`

The Streamlit Truth page supports node-type, sentiment, ReTruth, and minimum-count filters.
