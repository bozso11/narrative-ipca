"""narrative_ipca: production implementation of the BKS narrative asset pricing pipeline.

Stages (see DESIGN.md):

1. ``data``         input validation and calendar alignment
2. ``shocks``       attention shocks z_tau (BKS Section 3.1)
3. ``covariances``  kernel-weighted asset/narrative covariances (Eq. 6)
4. ``panel``        estimation panel c_{i,t-1} -> r_{i,t} (Eq. 7)
5. ``grouplasso``, ``sparse_ipca``   Sparse IPCA by ARLS (Eq. 8, 16)
6. ``tuning``       lambda / K selection (in-sample Sharpe, LOOCV)
7. ``wrapup``       A, latent states, impact vectors (Eq. 1, 5, 10-12)
8. ``oos``          expanding-window out-of-sample factors and MVE (Section 4.2)
9. ``evaluation``   R2, Sharpe, pricing tests, placebo test
10. ``pipeline``    orchestration; ``simulation`` + ``harness`` for validation.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import config, types  # noqa: F401  (light modules; heavy ones import lazily)

__all__ = ["config", "types", "__version__"]
