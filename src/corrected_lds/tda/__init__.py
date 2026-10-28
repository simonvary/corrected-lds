"""TDA score producers (Tier 2, GPU): TRAK (``trak.py``), EK-FAC influence
(``ekfac.py``), TracInCP (``tracin.py``) and SOURCE (``source.py``, on kronfluence's
EK-FAC factors).

TRAK and EK-FAC wrap ``traker==0.3.2`` and ``kronfluence==1.0.1``, exact pins because other
versions give different tau; TracIn implements its paper's Eq. 1 directly (the official
code is TF2 notebooks) on traker's projectors, under the same pin.  All write one
``tauq_*.npz`` schema, stamped with the pairing identity that
:func:`corrected_lds.banks.artifacts.predictions` enforces.  Modules import torch at module
level: import them only with the ``[gpu]`` extra installed.
"""
