"""MARITRACE application entry point."""
import streamlit as st

st.set_page_config(
    page_title="MARITRACE · Maritime intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.dashboard_v2 import render_dashboard

render_dashboard()
