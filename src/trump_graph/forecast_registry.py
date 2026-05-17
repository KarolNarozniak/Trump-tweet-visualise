from __future__ import annotations

from typing import Final


FORECAST_MODEL_REGISTRY: Final[tuple[dict[str, str], ...]] = (
    {
        "key": "tgn",
        "label": "TGN",
        "name": "Temporal Graph Network",
        "category": "Continuous-time event model",
        "paper_url": "https://arxiv.org/abs/2006.10637",
        "impl_url": "https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.models.TGNMemory.html",
        "desc": "Memory-based temporal message passing over timestamped events.",
        "family": "neural-temporal-graph",
    },
    {
        "key": "evolvegcn",
        "label": "EvolveGCN-H",
        "name": "Evolving Graph Convolutional Networks",
        "category": "Snapshot recurrent graph convolution",
        "paper_url": "https://ojs.aaai.org/index.php/AAAI/article/view/5984",
        "impl_url": "https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html",
        "desc": "Evolves GCN parameters through time with recurrent dynamics.",
        "family": "neural-temporal-graph",
    },
    {
        "key": "gconvgru",
        "label": "GConvGRU",
        "name": "Graph Convolutional Recurrent Network",
        "category": "Snapshot graph recurrent sequence model",
        "paper_url": "https://arxiv.org/abs/1612.07659",
        "impl_url": "https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html",
        "desc": "Combines graph convolution and GRU for temporal graph snapshots.",
        "family": "neural-temporal-graph",
    },
)

FORECAST_MODEL_BY_KEY: Final[dict[str, dict[str, str]]] = {
    str(model["key"]): model for model in FORECAST_MODEL_REGISTRY
}
SUPPORTED_FORECAST_MODEL_KEYS: Final[tuple[str, ...]] = tuple(FORECAST_MODEL_BY_KEY.keys())
