"""The "How to read" guide toggle of the lab dashboard (DESIGN.md G.9, D91).

Explanations use two elements: Streamlit's ``help=`` tooltip (the "?" of one tile, widget, button or table
column) and the guide toggle drawn here, a collapsed compact expander labelled in blue after a help icon, right
under the chart, table, row of tiles or note it explains. ``_ui`` builds the texts and stays free of Streamlit.
Never pass ``expanded``, ``icon`` (AppTest would parse the block as Status), ``help`` or ``wrap`` (the expander
has neither) or ``type="step"`` (1.64 only). ``key`` (``how_<site>``, unique on the page) keeps a guide open
across reruns until the reader closes it.

A separate module because ``trace_page`` cannot import ``app.py``, ``real_exposures`` imports Streamlit lazily,
and ``_ui`` stays pure.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

import _ui


def toggle(name: str, *, key: str) -> Any:
    """A collapsed guide toggle labelled ``name``, for a ``with`` block ("Why this method")."""
    return st.expander(_ui.guide_label(name), type="compact", key=key)


def guide(text: str | tuple[str, _ui.Bullets], *, key: str) -> None:
    """The "How to read" guide of one block.

    ``text`` is a :func:`_ui.how_to_read` text or a ``(lead, bullets)`` pair.
    The label is its lead without the colon; the body is its bullets as one
    caption.
    """
    if not isinstance(text, str):
        text = _ui.how_to_read(*text)
    label, body = _ui.guide_parts(text)
    with st.expander(label, type="compact", key=key):
        st.caption(body)
