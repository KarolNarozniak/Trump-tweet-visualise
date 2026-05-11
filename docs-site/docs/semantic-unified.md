# Unified Semantic Graph

This layer combines Twitter archive posts and Truth Social posts into one labeled temporal graph for consistent downstream modeling.

## Build Command

```bash
python -m trump_graph build-unified \
  --twitter-input "tweets_01-08-2021.csv" \
  --truth-input "truthsocial.posts[Trump-FROM-10-8-25].txt" \
  --out "data/processed_unified" \
  --semantic-backend hf \
  --device cuda
```

Deterministic smoke test:

```bash
python -m trump_graph build-unified \
  --twitter-input "tests/fixtures/sample_tweets.csv" \
  --truth-input "tests/fixtures/sample_truth_posts.csv" \
  --out "data/processed_unified" \
  --semantic-backend deterministic \
  --min-node-count 1
```

## Pipeline Behavior

- keeps both platforms in one normalized post table
- preserves weekly ISO bucketing
- runs topic labels, NER labels, sentiment labels, embeddings
- builds one stable global semantic graph
- writes sparse weekly deltas for node heat and edge accumulation

## Main Artifacts

- `unified_week_index.csv`
- `unified_weekly_summary.csv`
- `unified_posts_enriched.parquet`
- `unified_semantic_graph/animation_state.json`
- `unified_semantic_graph/node_catalog.csv`
- `unified_semantic_graph/edge_catalog.csv`
- `unified_embeddings/embeddings.npy`
- `unified_embeddings/index.csv`
- `unified_training/temporal_node_deltas.parquet`
- `unified_training/temporal_edge_deltas.parquet`

## Streamlit Endpoint

```text
http://localhost:3001/?page=semantic
```

The Semantic page supports:

- repost include/exclude
- sentiment filtering
- node type filtering
- minimum node frequency threshold
- weekly exports for posts and summary slices
