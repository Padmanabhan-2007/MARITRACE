"""Small shared Streamlit presentation components."""
from __future__ import annotations

import streamlit as st


def render_page_header(title: str, subtitle: str, eyebrow: str = "MARITRACE / INVESTIGATION SYSTEM") -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
        f'<div class="title">{title}</div><div class="lede">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )

