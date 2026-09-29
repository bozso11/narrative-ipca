"""Topic-exposure lab: simulated topics built from multi-asset prices (DESIGN.md Part G).

Library code only (no Streamlit imports); the dashboard is ``dashboard/app.py``.
Modules: ``config`` and ``types`` (contracts), ``reference`` and ``market``
(data, G.2), ``links`` and ``dgp`` (topics, links and simulation, G.3-G.5),
``direct`` and ``evaluate`` (direct exposure regression and out-of-sample
evaluation, G.7.1, G.8), ``bks`` (BKS Sparse IPCA, G.7.2), ``charts``
(Plotly figures) and ``session`` (cached orchestration).
"""

from narrative_ipca.exposure_lab import config, types

__all__ = ["config", "types"]
