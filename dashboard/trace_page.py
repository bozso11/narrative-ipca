"""Steps of the BKS trace page (DESIGN.md G.16, D90); placeholder until the page is drawn.

``render(ctx, ui)`` is called by ``app.bks_trace_page`` once a BKS fit of the current settings is cached. This
module cannot import ``app.py`` (that would re-run the script), so the widgets and helpers it needs come in ``ui``:
``control``, ``follow_control``, ``show_chart`` and ``bks_tiles``.
"""

from __future__ import annotations

from typing import Any


def render(ctx: dict[str, Any], ui: dict[str, Any]) -> None:
    import streamlit as st

    ui["bks_tiles"](ctx["res"])
    st.write(ctx["trace"].status_frame())
