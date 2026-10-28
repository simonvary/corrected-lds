"""The bank layer (Tier 1, torch-free): what a bank is, and loading/verifying one.

A **bank** is a released directory bundling one retrain pool with its tauq files: ``presets``
pins its identity, ``masks`` samples its subsets, ``artifacts`` loads and pairs its files, and
``hub`` fetches and pushes it (``fetch-banks``, ``push-bank``).  Never imports torch or the
GPU producers that fill a bank (``pools/`` the pool, ``tda/`` the tauqs).
"""
