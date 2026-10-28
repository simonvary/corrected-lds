"""PoolTask backends, one module per (data family, model family) setup.

Each supplies the model builder, per-cell trainer and margin readout that ``cli._make_task``
wires into a :class:`~corrected_lds.pools.runner.PoolTask`.  A new setup adds a recipe in
``presets.py``, a module here, a loader in ``pools/datasets.py`` (new data only), a preset
entry and a ``_make_task`` branch; never runner code.  Modules import torch at module level;
``imagenet_resnet18`` also needs torchvision (the ``[imagenet]`` extra).
"""
