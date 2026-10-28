"""Does the Pearson -> Spearman bridge bias the reconstructed estimator?  The CIFAR-10 and
QNLI numbers ``_fig5_panels`` reads, from ``bridge_estimators.json`` (no banks needed).

Figure 2(b,e)'s reconstructed route rebuilds each query's 5-seed Pearson from 2 seeds,
converts it with the arcsin bridge ``(6/pi) arcsin(r/2)`` and averages over queries; its
target rho_5 is the naive Spearman of 5-seed means.  Per method (E=1, figure 2's order):
blue square = reconstructed - naive on the full mask set (200 random 2-seed draws vs 40
random 5-seed subsets; 95% interval: delete-a-block jackknife over masks, 20 blocks);
grey band = +-1 SE of the reconstructed estimate (SD over the 2-seed draws).  The one
square outside its band, QNLI EK-FAC, is mostly the naive target's finite mask count
(decomposed in ``_bridge_estimators``).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared
from _harness import PLOTDATA_DIR, run


def compute(banks_root):
    with open(os.path.join(PLOTDATA_DIR, "bridge_estimators.json")) as fh:
        est = json.load(fh)
    panels = {}
    for bank, methods in _shared.B_METHODS.items():
        pq = est["panels"][bank]["per_query"]
        panels[bank] = [{"tda": tda, "label": label,
                         **{k: pq[tda][k] for k in ("D", "D_se", "se_estimator", "truth",
                                                    "conversion", "conversion_gaussian",
                                                    "n_draws", "n_truth_subsets")}}
                        for tda, label in methods]
    return {"banks": {b: est["banks"][b] for b in panels}, "panels": panels,
            "params": {"derived_from": ["bridge_estimators.json"], "E": "E1 replicas",
                       "interval": "95%, delete-a-block jackknife over masks (20 blocks)",
                       "band": "+-1 SE of the reconstructed estimate (SD over 2-seed "
                               "draws at the full mask set)"}}


if __name__ == "__main__":
    sys.exit(run("bridge_bias", compute))
