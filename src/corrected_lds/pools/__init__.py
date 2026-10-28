"""Retrain-pool builders (Tier 2, GPU).

``runner.py`` is torch-free orchestration.  ``train.py`` (shared SGD loop, margin readout) and
``tasks/`` (one PoolTask per (data, model) setup) import torch: install the ``[gpu]`` extra.
"""
