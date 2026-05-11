# Results Snapshot

This page captures an initial analysis snapshot from generated semantic artifacts.

## Core Volume

- weekly rows: **182**
- date range: **2022-02-14 to 2025-10-12**
- total posts processed: **18,475**
- original posts: **12,184**
- reposts/retruths: **6,291**

Post type split:

- original: **65.95%**
- repost/retruth: **34.05%**

Sentiment split:

- negative: **6,671** (**36.11%**)
- neutral: **5,502** (**29.78%**)
- positive: **6,302** (**34.11%**)

## Dominant Labels

Top labels by presence:

1. Media Communications
2. Law Justice Crime
3. Elections Campaigns
4. Health Public Health
5. Trump (person)

Observed behavior:

- `Media Communications` dominates both total count and weekly wins.
- The graph is mostly topic-driven, with entities significantly less frequent.

## Node-Type Mix

- topics: **25,844**
- persons: **4,612**
- locations: **1,500**
- mentions: **914**
- organizations: **463**
- hashtags: **11**

## Weekly Winner Pattern

Most weeks have `Media Communications` as top topic:

- Media Communications: **172 weeks**
- Law Justice Crime: **5 weeks**
- Endorsements Appointments: **3 weeks**
- Congress Governance: **1 week**
- Health Public Health: **1 week**

## Data Quality Notes

Some extracted entities are fragmented (`B`, `E`, `F`, `Iden`, `Kamal`), which indicates additional NER normalization is needed before final model training.

Planned improvements:

- stronger post-processing dictionary for entity canonicalization
- merge aliases (`Kamal` -> `Kamala Harris`, `Iden` -> `Biden`)
- stricter low-token cleanup for PER/ORG entities

## Year-Level Sentiment Trend

- 2022: 38.4% negative, 31.4% neutral, 30.2% positive
- 2023: 41.1% negative, 32.3% neutral, 26.6% positive
- 2024: 34.2% negative, 30.8% neutral, 35.0% positive
- 2025: 27.4% negative, 19.7% neutral, 53.0% positive

This snapshot is expected to evolve as the labeling cleanup and temporal model stack improve.
