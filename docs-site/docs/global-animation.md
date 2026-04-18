# Global Animation Model

The app renders a single persistent graph. Node positions do not change during playback.

## Node Behavior

- Nodes become visible once they appear.
- Nodes stay visible after first appearance.
- Weekly activity controls node heat color.
- Node size is based on global mention totals.

Size mapping:

```text
size = size_min + size_scale * sqrt(total_mentions / p99_total_mentions)
```

## Edge Behavior

- Edge appears when first co-mention occurs.
- Edge width grows cumulatively across weeks.
- Edge color highlights active-week interactions.
- Inactive edges remain white-tinted.

Width mapping:

```text
if C_t(e) == 0: hide edge
else: width = 0.9 + 7.0 * sqrt(C_t(e) / max_cumulative_edge)
```

## Weekly Transition

The renderer interpolates between week states over sub-frames so transitions are continuous rather than slideshow-like.

## Stability Guarantees

- deterministic node layout (`layout_seed`)
- fixed coordinates persisted in build artifacts
- no re-layout during playback
