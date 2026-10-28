"""Debiased, disattenuated LDS evaluation for training-data attribution.

The core imports only numpy/scipy and scores released retrain banks (Tier 1, CPU); the GPU
builders in ``pools/`` and ``tda/`` need the ``[gpu]`` extra and import torch lazily (Tier 2).
"""

from corrected_lds.banks.artifacts import (
    combine_replicas,
    concat_pool_tda,
    load_pool,
    load_pool_tda,
    load_tauq,
    predictions,
    predictions_tda,
    write_pool_tda,
    write_tauq,
)
from corrected_lds.estimator import (
    battery,
    corrected_pearson,
    pearson_to_spearman,
    reconstruct_rho_R,
    seed_noise_variance_centred,
    seed_noise_variance_stacked,
)
from corrected_lds.evaluate import evaluate_pool, evaluate_pool_tda
from corrected_lds.tda_noise import (
    battery_c,
    disattenuate,
    member_noise_ratio,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "battery",
    "battery_c",
    "combine_replicas",
    "concat_pool_tda",
    "corrected_pearson",
    "disattenuate",
    "evaluate_pool",
    "evaluate_pool_tda",
    "load_pool",
    "load_pool_tda",
    "load_tauq",
    "member_noise_ratio",
    "pearson_to_spearman",
    "predictions",
    "predictions_tda",
    "reconstruct_rho_R",
    "seed_noise_variance_centred",
    "seed_noise_variance_stacked",
    "write_pool_tda",
    "write_tauq",
]
