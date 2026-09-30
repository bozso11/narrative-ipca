"""Topic-sensitivity lab: simulated topics built from multi-asset prices (DESIGN.md Part G).

Library code only (no Streamlit imports); the dashboard is ``dashboard/app.py``.
Modules: ``config`` and ``types`` (contracts), ``reference`` and ``market``
(data, G.2), ``links`` and ``dgp`` (topics, links and simulation, G.3-G.5),
``direct`` and ``evaluate`` (direct sensitivity regression and out-of-sample
evaluation, G.7.1, G.8), ``bks`` (BKS Sparse IPCA, G.7.2), ``compare``
(method comparison, G.15), ``charts`` (Plotly figures) and ``session``
(cached orchestration).

Terminology
-----------
The **topic sensitivity** ``b_kn`` (the matrix ``B``, topics x assets) is the
expected return response of asset ``n`` to a one-standard-deviation attention
shock in topic ``k``, with the other topics' shocks held fixed. It is the
coefficient in the regression of the asset's return on all topics' attention
shocks at once::

    r_{n,t+l} = a_n + sum_k b_kn s_{k,t} + e_{n,t+l}

where ``r_{n,t+l}`` is asset ``n``'s return on day ``t+l``, ``s_{k,t}`` is
topic ``k``'s attention shock on day ``t`` (attention minus its mean over the
previous ``w`` days, divided by its standard deviation), ``a_n`` is an
intercept, ``e`` is the part not explained by topics and ``l`` is the lead
(0 = same day, 1 = next day). It is not a position size or dollar exposure.
The lab has three versions: the set sensitivity ``W`` (set with the link map
and the "betas"), the true sensitivity ``B_true`` (the population value of the
simulation, including spillovers) and the estimated sensitivity ``B_hat``
(each method's training-window estimate).

In code, "exposure" means topic sensitivity: the package name
``exposure_lab``, ``ExposureConfig``, ``exposure_heatmap``, ``B_hat`` and the
"exposures" in docstrings and variable names keep the older word. The
user-facing text was renamed on 2026-09-30, because "exposure" is easily read
as a dollar exposure to the asset.
"""

from narrative_ipca.exposure_lab import config, types

__all__ = ["config", "types"]
