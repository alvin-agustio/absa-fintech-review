"""
Two-page Streamlit app:
1) All-in-one sentiment observatory (general -> detail)
2) Preprocessing funnel and data quality diagnostics
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from datetime import date, datetime, timedelta
import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import APPS, DATA_PROCESSED, DATA_RAW
from src.dashboard.analytics import (
    ASPECT_ORDER,
    compute_kpis,
    hydrate_scope,
    wide_review_frame,
)
from src.dashboard.aspect_taxonomy import (
    GENERAL_ISSUE_LABEL,
    aspect_display_name,
    aspect_presence_details,
    assign_issue_label,
)
from src.dashboard.live import run_live_analysis
from src.dashboard.registry import build_model_registry, default_model_row
from src.dashboard.research import load_gold_summary, load_gold_subset, load_weak_overview
from src.dashboard.storage import DashboardStore

try:
    from src.dashboard.summary_rules import build_summary_payload as dashboard_summary_payload
except ImportError:
    dashboard_summary_payload = None


st.set_page_config(
    page_title="Fintech Sentiment Observatory",
    page_icon="FSO",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
:root {
  --bg: #f3f1ea;
  --surface: #fdfbf6;
  --ink: #1f211e;
    --muted: #50544e;
  --border: #d8d2c6;
  --service: #2f7f52;
  --trust: #1b7286;
  --risk: #b44e34;
  --accent: #2a2f45;
}
.stApp {
  background: radial-gradient(circle at 0% 0%, #faf8f1 0%, var(--bg) 40%, #ebe5d8 100%);
  color: var(--ink);
  font-family: 'Plus Jakarta Sans', sans-serif;
}
[data-testid="stAppViewContainer"],
[data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"],
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] {
    color: var(--ink) !important;
}
[data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
}
[data-baseweb="select"],
[data-baseweb="select"] * {
    color: var(--ink) !important;
}
[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
}
[data-baseweb="select"] input,
[data-baseweb="select"] span,
[data-baseweb="select"] div {
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
}
[role="listbox"] {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
}
[role="option"] {
    color: var(--ink) !important;
}
[role="option"][aria-selected="true"] {
    background: #eef2ff !important;
}
[data-testid="stDateInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
}
[data-testid="stDateInput"] svg,
[data-baseweb="select"] svg {
    fill: #3b3f4a !important;
}
[data-testid="stSlider"] * {
    color: var(--ink) !important;
}
[role="radiogroup"],
[role="radiogroup"] * {
    color: var(--ink) !important;
}

/* Sidebar-specific high-contrast palette */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #222533 0%, #1b1e29 100%) !important;
}
[data-testid="stSidebar"] * {
    color: #eef1f7 !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] .stCaption {
    color: #c6ccda !important;
}
[data-testid="stSidebar"] [role="radiogroup"] *,
[data-testid="stSidebar"] label {
    color: #eef1f7 !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] input:checked + div {
    box-shadow: inset 0 0 0 5px #ff4d4f !important;
}

label, .stRadio label, .stSelectbox label, .stDateInput label {
    color: var(--muted) !important;
}
[data-testid="stDataFrame"] * {
    color: var(--ink) !important;
}
.block-container {
  max-width: 1560px;
  padding-top: 2.4rem;
  padding-bottom: 2.7rem;
}
[data-testid="stAppViewBlockContainer"] {
  padding-top: 2.7rem;
}
h1, h2, h3 {
  font-family: 'Outfit', sans-serif;
  letter-spacing: -0.01em;
}
.hero {
  background:
    radial-gradient(circle at top right, rgba(72, 85, 130, 0.08), transparent 45%),
    linear-gradient(180deg, rgba(252, 250, 246, 0.98), rgba(250, 247, 240, 0.95));
  border: 1px solid rgba(216, 210, 198, 0.95);
  border-radius: 26px;
  padding: 1.2rem 1.35rem;
  margin-bottom: 1.1rem;
  box-shadow: 0 12px 30px rgba(30, 33, 38, 0.06);
}
.eyebrow {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.13em;
  font-size: 0.74rem;
  font-weight: 700;
}
.section-intro {
  color: var(--muted);
  font-size: 0.92rem;
  margin-top: 0.35rem;
}
[data-testid="stMetric"] {
  background: rgba(253, 251, 246, 0.86);
  border: 1px solid rgba(216, 210, 198, 0.9);
  border-radius: 14px;
  padding: 0.65rem 0.75rem;
}
[data-testid="stMetric"] label {
  color: var(--muted) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(253, 251, 246, 0.28);
  border: 1px solid transparent !important;
  border-radius: 18px !important;
  box-shadow: none;
  margin-bottom: 0.7rem;
}
[data-testid="stExpander"] {
  background: rgba(253, 251, 246, 0.34);
  border: 1px solid transparent !important;
  border-radius: 18px !important;
  box-shadow: none;
  margin-top: 0.35rem;
  margin-bottom: 0.55rem;
}
[data-testid="stExpanderDetails"] {
  padding-top: 0.55rem;
}
[data-testid="stDataFrame"],
[data-testid="stTable"] {
  border: 1px solid transparent !important;
  box-shadow: none !important;
}
[data-testid="stDataFrame"] {
  border-radius: 12px;
}
.insight-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.82), rgba(252,248,241,0.92));
  border: 1px solid rgba(216, 210, 198, 0.95);
  border-radius: 20px;
  padding: 1rem 1rem 0.95rem 1rem;
  min-height: 236px;
  box-shadow: 0 10px 22px rgba(31, 33, 30, 0.06);
}
.insight-card h4 {
  margin: 0 0 0.4rem 0;
  font-family: 'Outfit', sans-serif;
  font-size: 1.15rem;
}
.card-title {
  margin: 0 0 0.4rem 0;
  font-family: 'Outfit', sans-serif;
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--ink);
}
.eyebrow-mini {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.68rem;
  font-weight: 700;
  margin-bottom: 0.45rem;
}
.metric-line {
  margin: 0.34rem 0;
  line-height: 1.55;
  color: var(--ink);
  font-size: 0.92rem;
}
.metric-label {
  color: var(--muted);
  font-weight: 600;
}
.pill {
  display: inline-block;
  padding: 0.18rem 0.48rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 700;
  border: 1px solid transparent;
  margin: 0 0.2rem 0.2rem 0;
}
.pill-risk {
  background: rgba(180, 78, 52, 0.1);
  color: #933b26;
  border-color: rgba(180, 78, 52, 0.18);
}
.pill-trust {
  background: rgba(27, 114, 134, 0.1);
  color: #0f5a6b;
  border-color: rgba(27, 114, 134, 0.18);
}
.pill-service {
  background: rgba(47, 127, 82, 0.1);
  color: #256342;
  border-color: rgba(47, 127, 82, 0.18);
}
.pill-positive {
  background: rgba(47, 127, 82, 0.12);
  color: #256342;
  border-color: rgba(47, 127, 82, 0.22);
}
.pill-negative {
  background: rgba(180, 78, 52, 0.12);
  color: #933b26;
  border-color: rgba(180, 78, 52, 0.22);
}
.pill-neutral {
  background: rgba(81, 84, 78, 0.08);
  color: #50544e;
  border-color: rgba(81, 84, 78, 0.12);
}
.issue-card {
  background: rgba(255, 255, 255, 0.78);
  border: 1px solid rgba(216, 210, 198, 0.88);
  border-radius: 18px;
  padding: 0.95rem 1rem;
  min-height: 296px;
  margin: 0.2rem 0 0.7rem 0;
}
.issue-card .card-title {
  margin-bottom: 0.3rem;
  font-size: 1.1rem;
}
.issue-row {
  padding: 0.58rem 0 0.6rem 0;
  border-top: 1px solid rgba(216, 210, 198, 0.65);
}
.issue-row:first-of-type {
  border-top: none;
  padding-top: 0.15rem;
}
.issue-title {
  font-weight: 700;
  color: var(--ink);
  margin-bottom: 0.22rem;
}
.issue-meta {
  color: var(--ink);
  line-height: 1.5;
  font-size: 0.9rem;
}
.issue-submeta {
  color: var(--muted);
  line-height: 1.45;
  font-size: 0.8rem;
  margin-top: 0.12rem;
}
.trust-note {
  background: rgba(42, 47, 69, 0.06);
  border: 1px dashed rgba(42, 47, 69, 0.16);
  border-radius: 14px;
  padding: 0.68rem 0.78rem;
  color: var(--muted);
  font-size: 0.81rem;
  line-height: 1.45;
}
.issue-card .trust-note {
  margin-bottom: 0.3rem;
}
.evidence-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.84), rgba(251,247,239,0.96));
  border: 1px solid rgba(216, 210, 198, 0.95);
  border-radius: 20px;
  padding: 1rem 1rem 0.95rem 1rem;
  min-height: 320px;
  box-shadow: 0 12px 24px rgba(31, 33, 30, 0.05);
  display: flex;
  flex-direction: column;
  gap: 0.7rem;
}
.evidence-card-risk {
  border-top: 4px solid rgba(180, 78, 52, 0.58);
}
.evidence-card-trust {
  border-top: 4px solid rgba(27, 114, 134, 0.58);
}
.evidence-card-service {
  border-top: 4px solid rgba(47, 127, 82, 0.58);
}
.evidence-meta {
  color: var(--muted);
  font-size: 0.8rem;
  line-height: 1.45;
}
.evidence-pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.3rem;
}
.evidence-signal {
  background: rgba(42, 47, 69, 0.05);
  border: 1px dashed rgba(42, 47, 69, 0.14);
  border-radius: 14px;
  padding: 0.66rem 0.74rem;
  color: var(--muted);
  font-size: 0.8rem;
  line-height: 1.45;
}
.evidence-rationale {
  color: var(--ink);
  font-size: 0.92rem;
  line-height: 1.55;
}
.evidence-quote {
  margin-top: auto;
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(216, 210, 198, 0.74);
  border-left: 4px solid rgba(42, 47, 69, 0.38);
  border-radius: 14px;
  padding: 0.82rem 0.9rem;
  color: var(--ink);
  font-size: 0.92rem;
  line-height: 1.58;
}
.scope-strip {
  background: linear-gradient(180deg, rgba(255,255,255,0.82), rgba(252,248,241,0.92));
  border: 1px solid rgba(216, 210, 198, 0.95);
  border-radius: 18px;
  padding: 1rem 1rem 0.8rem 1rem;
}
.scope-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 1rem;
  margin-top: 0.8rem;
}
.scope-cell {
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(216, 210, 198, 0.82);
  border-radius: 14px;
  padding: 0.8rem 0.85rem;
}
.scope-label {
  color: var(--muted);
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-weight: 700;
  margin-bottom: 0.3rem;
}
.scope-value {
  color: var(--ink);
  font-size: 1.05rem;
  font-weight: 700;
  line-height: 1.2;
  word-break: break-word;
}
.scope-subline {
  margin-top: 0.75rem;
  color: var(--muted);
  font-size: 0.84rem;
  line-height: 1.5;
}
.health-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1.1rem;
  margin-top: 0.9rem;
  margin-bottom: 0.35rem;
}
.health-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.82), rgba(252,248,241,0.92));
  border: 1px solid rgba(216, 210, 198, 0.95);
  border-radius: 20px;
  padding: 1rem 1rem 0.95rem 1rem;
  box-shadow: 0 10px 22px rgba(31, 33, 30, 0.05);
}
.health-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.75rem;
  margin-bottom: 0.75rem;
}
.health-name {
  font-family: 'Outfit', sans-serif;
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--ink);
}
.health-meta {
  text-align: right;
}
.health-balance {
  color: var(--muted);
  font-size: 0.84rem;
  line-height: 1.5;
}
.health-bars {
  display: grid;
  gap: 0.6rem;
}
.health-row {
  display: grid;
  grid-template-columns: 68px 1fr 42px;
  align-items: center;
  gap: 0.6rem;
}
.health-row-label {
  color: var(--muted);
  font-size: 0.82rem;
  font-weight: 600;
}
.health-track {
  height: 8px;
  border-radius: 999px;
  background: rgba(81, 84, 78, 0.08);
  overflow: hidden;
}
.health-fill-positive {
  height: 100%;
  background: linear-gradient(90deg, #7ecb9a, #2f7f52);
}
.health-fill-neutral {
  height: 100%;
  background: linear-gradient(90deg, #bec2c8, #8c8f95);
}
.health-fill-negative {
  height: 100%;
  background: linear-gradient(90deg, #d88c74, #b44e34);
}
.health-row-value {
  color: var(--ink);
  font-size: 0.82rem;
  font-weight: 700;
  text-align: right;
}
.health-footer {
  margin-top: 0.75rem;
}
.sentiment-stack {
  display: flex;
  width: 100%;
  height: 14px;
  overflow: hidden;
  border-radius: 999px;
  background: rgba(81, 84, 78, 0.08);
  border: 1px solid rgba(216, 210, 198, 0.7);
}
.sentiment-segment {
  height: 100%;
}
.sentiment-segment-positive {
  background: linear-gradient(90deg, #8dd6a6, #2f7f52);
}
.sentiment-segment-neutral {
  background: linear-gradient(90deg, #c8ccd2, #8c8f95);
}
.sentiment-segment-negative {
  background: linear-gradient(90deg, #e19a86, #b44e34);
}
.sentiment-legend {
  display: grid;
  gap: 0.45rem;
  margin-top: 0.7rem;
}
.sentiment-legend-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
}
.sentiment-legend-label {
  color: var(--muted);
  font-size: 0.82rem;
  font-weight: 600;
}
.sentiment-legend-value {
  color: var(--ink);
  font-size: 0.84rem;
  font-weight: 700;
}
.summary-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.85rem;
  margin-top: 0.72rem;
  margin-bottom: 0.25rem;
}
.summary-card {
  background: rgba(255,255,255,0.68);
  border: 1px solid rgba(216, 210, 198, 0.88);
  border-radius: 14px;
  padding: 0.82rem 0.88rem;
}
.summary-title {
  color: var(--muted);
  font-size: 0.77rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-weight: 700;
  margin-bottom: 0.28rem;
}
.summary-value {
  color: var(--ink);
  font-size: 1.15rem;
  font-weight: 700;
}
.diagnosis-shell {
  display: grid;
  gap: 0.75rem;
}
.diagnosis-summary {
  background: linear-gradient(180deg, rgba(255,255,255,0.82), rgba(252,248,241,0.92));
  border: 1px solid rgba(216, 210, 198, 0.95);
  border-radius: 18px;
  padding: 0.92rem 0.95rem 0.9rem 0.95rem;
  min-height: 272px;
  margin-bottom: 0.7rem;
  display: flex;
  flex-direction: column;
  gap: 0.58rem;
}
.diagnosis-lead {
  color: var(--ink);
  font-size: 0.93rem;
  line-height: 1.55;
  font-weight: 600;
  margin-top: 0.12rem;
  min-height: 4.7rem;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.diagnosis-rows {
  display: grid;
  gap: 0.44rem;
  margin-top: 0.1rem;
}
.diagnosis-row {
  display: grid;
  grid-template-columns: 132px minmax(0, 1fr);
  gap: 0.7rem;
  align-items: center;
  min-height: 3.05rem;
  padding-top: 0.44rem;
  border-top: 1px solid rgba(216, 210, 198, 0.55);
}
.diagnosis-row:first-child {
  padding-top: 0.05rem;
  border-top: none;
}
.diagnosis-row-label {
  color: var(--muted);
  font-size: 0.77rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
  white-space: nowrap;
}
.diagnosis-row-value {
  color: var(--ink);
  font-size: 0.87rem;
  line-height: 1.5;
  min-width: 0;
  min-height: 2.45rem;
  display: flex;
  align-items: center;
}
.diagnosis-row-value-text {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.diagnosis-distribution {
  display: grid;
  gap: 0.2rem;
  min-height: 2.45rem;
}
.diagnosis-dist-line {
  color: var(--ink);
  font-size: 0.84rem;
  line-height: 1.4;
  font-weight: 600;
  padding: 0.14rem 0.5rem;
  border-radius: 999px;
  width: fit-content;
}
.diagnosis-dist-line-negative {
  color: #8f4734;
  background: rgba(180, 78, 52, 0.10);
  border: 1px solid rgba(180, 78, 52, 0.16);
}
.diagnosis-dist-line-positive {
  color: #2c6a47;
  background: rgba(47, 127, 82, 0.10);
  border: 1px solid rgba(47, 127, 82, 0.16);
}
.diagnosis-detail-stack {
  display: grid;
  gap: 0.32rem;
  margin-top: 0.15rem;
}
.diagnosis-copy {
  color: var(--muted);
  font-size: 0.88rem;
  line-height: 1.55;
  margin-top: 0.45rem;
}
.example-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.9rem;
  margin-top: 0.45rem;
}
.example-section {
  display: grid;
  gap: 0.65rem;
  margin-top: 0.7rem;
}
.example-section:first-child {
  margin-top: 0.05rem;
}
.example-section-title {
  font-family: 'Outfit', sans-serif;
  font-size: 0.96rem;
  font-weight: 700;
  color: var(--ink);
  margin: 0.05rem 0 0 0;
}
.example-section-note {
  color: var(--muted);
  font-size: 0.82rem;
  line-height: 1.45;
  margin: 0.06rem 0 0.55rem 0;
}
.example-column {
  display: grid;
  gap: 0.6rem;
}
.example-column-title {
  font-family: 'Outfit', sans-serif;
  font-size: 0.96rem;
  font-weight: 700;
  color: var(--ink);
}
.example-card {
  background: rgba(255,255,255,0.8);
  border: 1px solid rgba(216, 210, 198, 0.9);
  border-radius: 16px;
  padding: 0.8rem 0.85rem;
  box-shadow: 0 8px 18px rgba(31, 33, 30, 0.04);
}
.example-card-positive {
  border-top: 3px solid rgba(47, 127, 82, 0.65);
}
.example-card-neutral {
  border-top: 3px solid rgba(140, 143, 149, 0.7);
}
.example-card-negative {
  border-top: 3px solid rgba(180, 78, 52, 0.65);
}
.example-meta {
  color: var(--muted);
  font-size: 0.78rem;
  line-height: 1.45;
  margin-bottom: 0.45rem;
}
.example-quote {
  color: var(--ink);
  font-size: 0.88rem;
  line-height: 1.55;
}
.example-native-card {
  background: rgba(255,255,255,0.82);
  border: 1px solid rgba(216, 210, 198, 0.9);
  border-radius: 16px;
  padding: 0.82rem 0.9rem;
  box-shadow: 0 8px 18px rgba(31, 33, 30, 0.04);
  margin: 0 0 0.72rem 0;
}
.example-native-card-positive {
  border-top: 3px solid rgba(47, 127, 82, 0.68);
}
.example-native-card-neutral {
  border-top: 3px solid rgba(140, 143, 149, 0.72);
}
.example-native-card-negative {
  border-top: 3px solid rgba(180, 78, 52, 0.68);
}
.example-native-meta {
  color: var(--muted);
  font-size: 0.78rem;
  line-height: 1.45;
  margin-bottom: 0.45rem;
}
.example-native-text {
  color: var(--ink);
  font-size: 0.9rem;
  line-height: 1.55;
}
.conclusion-stack {
  display: grid;
  gap: 0.72rem;
}
.conclusion-grid-2 {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.84rem;
}
.conclusion-grid-1 {
  display: grid;
  gap: 0.72rem;
}
.conclusion-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.82), rgba(252,248,241,0.92));
  border: 1px solid rgba(216, 210, 198, 0.95);
  border-radius: 18px;
  padding: 0.98rem 1.05rem;
  margin: 0;
  min-height: 204px;
  display: flex;
  flex-direction: column;
}
.conclusion-card-lead {
  border-top: 5px solid rgba(42, 47, 69, 0.28);
}
.conclusion-card-app {
  border-top: 4px solid rgba(42, 47, 69, 0.22);
}
.conclusion-card-focus {
  border-top: 4px solid rgba(180, 78, 52, 0.7);
}
.conclusion-card-close {
  border-top: 4px solid rgba(140, 143, 149, 0.72);
}
.conclusion-card-positive {
  border-top: 4px solid rgba(47, 127, 82, 0.7);
}
.conclusion-card-neutral {
  border-top: 4px solid rgba(140, 143, 149, 0.72);
}
.conclusion-card-negative {
  border-top: 4px solid rgba(180, 78, 52, 0.7);
}
.conclusion-title {
  color: var(--ink);
  font-family: 'Outfit', sans-serif;
  font-size: 1.03rem;
  font-weight: 700;
  margin: 0.12rem 0 0.32rem 0;
}
.conclusion-copy {
  color: var(--ink);
  font-size: 0.92rem;
  line-height: 1.58;
  margin-top: 0.22rem;
}
.conclusion-lead {
  margin-top: 0.2rem;
  color: var(--ink);
  font-size: 0.94rem;
  line-height: 1.56;
  font-weight: 600;
}
.conclusion-subline {
  color: var(--muted);
  font-size: 0.84rem;
  line-height: 1.52;
  margin-top: 0.34rem;
}
.conclusion-evidence-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.38rem;
  margin-top: 0.72rem;
  margin-top: auto;
  padding-top: 0.72rem;
}
.conclusion-chip {
  display: inline-flex;
  align-items: center;
  padding: 0.2rem 0.52rem;
  border-radius: 999px;
  background: rgba(42, 47, 69, 0.05);
  border: 1px solid rgba(42, 47, 69, 0.12);
  color: var(--muted);
  font-size: 0.76rem;
  line-height: 1.2;
  font-weight: 600;
}
.section-spacer-sm {
  height: 0.12rem;
}
.section-spacer-md {
  height: 0.6rem;
}
@media (max-width: 1100px) {
  .scope-grid, .health-grid, .summary-grid, .example-grid {
    grid-template-columns: 1fr 1fr;
  }
}
@media (max-width: 760px) {
  .scope-grid, .health-grid, .summary-grid, .example-grid {
    grid-template-columns: 1fr;
  }
  .health-row {
    grid-template-columns: 62px 1fr 38px;
  }
}
</style>
""",
    unsafe_allow_html=True,
)


ASPECT_COLOR_MAP = {"service": "#2f7f52", "trust": "#1b7286", "risk": "#b44e34"}
SENTIMENT_COLOR_MAP = {"Positive": "#2f7f52", "Neutral": "#8c8f95", "Negative": "#b44e34"}
ASPECT_LABEL_MAP = {"risk": "Risk", "trust": "Trust", "service": "Service"}
STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "karena", "pada", "nya", "saya", "aku",
    "kami", "kita", "anda", "kalian", "the", "a", "is", "are", "app", "aplikasi", "kredivo", "akulaku", "aja",
    "banget", "sangat", "lebih", "jadi", "udah", "ga", "gak", "tidak", "nggak", "yg", "tp", "kalo", "kalau",
    "atau", "dalam", "juga", "sih", "dong", "lah", "nih", "kok", "karna", "sudah", "belum",
}


def chart_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Plus Jakarta Sans", "color": "#1f211e"},
        margin=dict(l=20, r=20, t=30, b=20),
        legend_title_text="",
    )
    return fig


@st.cache_resource(show_spinner=False)
def get_store() -> DashboardStore:
    store = DashboardStore()
    store.initialize()
    return store


@st.cache_data(show_spinner=False)
def get_registry() -> pd.DataFrame:
    return build_model_registry()


@st.cache_data(show_spinner=False)
def get_gold_summary_data() -> dict:
    return load_gold_summary()


@st.cache_data(show_spinner=False)
def get_gold_subset_data() -> pd.DataFrame:
    return load_gold_subset()


@st.cache_data(show_spinner=False)
def get_weak_overview_data() -> pd.DataFrame:
    return load_weak_overview()


@st.cache_resource(show_spinner=True)
def get_predictor(model_dir: str):
    from src.inference import ABSAPredictor

    return ABSAPredictor(model_dir)


def select_period(mode: str, start_custom: date, end_custom: date) -> tuple[date, date]:
    today = datetime.today().date()
    if mode == "7d":
        return today - timedelta(days=6), today
    if mode == "30d":
        return today - timedelta(days=29), today
    if mode == "90d":
        return today - timedelta(days=89), today
    return start_custom, end_custom


def format_jobs(jobs_df: pd.DataFrame, registry_df: pd.DataFrame) -> pd.DataFrame:
    if jobs_df.empty:
        return jobs_df

    display = jobs_df.copy()
    if {"model_id", "display_name"}.issubset(registry_df.columns):
        display = display.merge(registry_df[["model_id", "display_name"]], on="model_id", how="left")
    else:
        display["display_name"] = display["model_id"]

    display["label"] = display.apply(
        lambda row: (
            f"{row['job_id']} | {row['app_name']} | {row['date_from']} -> "
            f"{row['date_to']} | {coalesce_text(row.get('display_name'), row['model_id'])} | "
            f"limit={human_limit_label(row.get('review_limit'))}"
        ),
        axis=1,
    )
    return display.sort_values("fetched_at", ascending=False).reset_index(drop=True)


def coalesce_text(value, fallback: str = "-") -> str:
    if value is None:
        return str(fallback)
    if isinstance(value, str) and value == "":
        return str(fallback)
    try:
        if bool(pd.isna(value)):
            return str(fallback)
    except TypeError:
        pass
    return str(value)


def human_limit_label(value) -> str:
    if value is None:
        return "all"
    if isinstance(value, str) and value in ("", "None"):
        return "all"
    try:
        if bool(pd.isna(value)):
            return "all"
    except TypeError:
        pass
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return "all"


def benchmark_rows_available(compare_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if compare_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    gold_ready = compare_df[compare_df["gold_f1_macro"].notna()].copy()
    weak_ready = compare_df[compare_df["weak_f1_macro"].notna()].copy()
    return gold_ready, weak_ready


def aspect_score_table(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(
            columns=[
                "aspect",
                "positive_share",
                "neutral_share",
                "negative_share",
                "dominant_sentiment",
                "dominant_share",
                "balance_note",
            ]
        )

    pivot = (
        long_df.groupby(["aspect", "pred_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .pivot(index="aspect", columns="pred_label", values="count")
        .fillna(0)
    )
    for label in ["Positive", "Neutral", "Negative"]:
        if label not in pivot.columns:
            pivot[label] = 0

    total = pivot["Positive"] + pivot["Neutral"] + pivot["Negative"]
    pos_share = (pivot["Positive"] / total).fillna(0)
    neg_share = (pivot["Negative"] / total).fillna(0)
    neu_share = (pivot["Neutral"] / total).fillna(0)
    dominant_sentiment = pd.DataFrame(
        {"Positive": pos_share, "Neutral": neu_share, "Negative": neg_share}
    ).idxmax(axis=1)
    dominant_share = pd.DataFrame(
        {"Positive": pos_share, "Neutral": neu_share, "Negative": neg_share}
    ).max(axis=1)

    def balance_note(positive: float, neutral: float, negative: float) -> str:
        top_share = max(positive, neutral, negative)
        if top_share < 0.45:
            return "Komposisi masih campuran"
        if negative >= positive and negative >= neutral:
            return "Negatif paling dominan"
        if positive >= negative and positive >= neutral:
            return "Positif paling dominan"
        return "Netral paling dominan"

    out = pd.DataFrame(
        {
            "aspect": pivot.index,
            "positive_share": (pos_share * 100).round(0).astype(int),
            "neutral_share": (neu_share * 100).round(0).astype(int),
            "negative_share": (neg_share * 100).round(0).astype(int),
            "dominant_sentiment": dominant_sentiment,
            "dominant_share": (dominant_share * 100).round(0).astype(int),
        }
    ).reset_index(drop=True)
    out["balance_note"] = [
        balance_note(float(pos), float(neu), float(neg))
        for pos, neu, neg in zip(pos_share.tolist(), neu_share.tolist(), neg_share.tolist())
    ]
    out["aspect"] = pd.Categorical(out["aspect"], categories=["risk", "trust", "service"], ordered=True)
    return out.sort_values("aspect").reset_index(drop=True)


def keyword_clusters(long_df: pd.DataFrame, aspect: str, n: int = 8) -> list[tuple[str, int]]:
    if long_df.empty:
        return []
    subset = long_df[(long_df["aspect"] == aspect) & (long_df["pred_label"] == "Negative")]
    if subset.empty:
        return []

    words: list[str] = []
    for text in subset["review_text_clean"].astype(str):
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
        words.extend([token for token in tokens if token not in STOPWORDS])
    return Counter(words).most_common(n)


def word_frequency_frame(wide_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if wide_df.empty:
        return pd.DataFrame(columns=["word", "count"])

    words: list[str] = []
    for text in wide_df["review_text_clean"].fillna("").astype(str):
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
        words.extend([token for token in tokens if token not in STOPWORDS])

    if not words:
        return pd.DataFrame(columns=["word", "count"])

    freq = Counter(words).most_common(top_n)
    return pd.DataFrame(freq, columns=["word", "count"])


def similar_comment_pairs(wide_df: pd.DataFrame, top_k: int = 12, max_reviews: int = 250) -> pd.DataFrame:
    if wide_df.empty:
        return pd.DataFrame()

    sample = wide_df[["app_name", "review_text_raw", "review_text_clean"]].dropna(subset=["review_text_clean"]).copy()
    sample = sample[sample["review_text_clean"].astype(str).str.len() >= 10].head(max_reviews).reset_index(drop=True)
    if len(sample) < 2:
        return pd.DataFrame()

    import importlib.util

    use_sklearn = importlib.util.find_spec("sklearn") is not None
    if use_sklearn:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        mat = vec.fit_transform(sample["review_text_clean"].astype(str).tolist())
        sim = cosine_similarity(mat)

        pairs = []
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                score = float(sim[i, j])
                if score >= 0.5:
                    pairs.append(
                        {
                            "similarity": round(score, 3),
                            "app_a": sample.iloc[i]["app_name"],
                            "comment_a": str(sample.iloc[i]["review_text_raw"]),
                            "app_b": sample.iloc[j]["app_name"],
                            "comment_b": str(sample.iloc[j]["review_text_raw"]),
                        }
                    )
        if not pairs:
            return pd.DataFrame()
        return pd.DataFrame(pairs).sort_values("similarity", ascending=False).head(top_k).reset_index(drop=True)

    token_sets = []
    for text in sample["review_text_clean"].astype(str):
        tokens = [t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower()) if t not in STOPWORDS]
        token_sets.append(set(tokens))

    pairs = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            union = token_sets[i].union(token_sets[j])
            if not union:
                continue
            score = len(token_sets[i].intersection(token_sets[j])) / len(union)
            if score >= 0.4:
                pairs.append(
                    {
                        "similarity": round(float(score), 3),
                        "app_a": sample.iloc[i]["app_name"],
                        "comment_a": str(sample.iloc[i]["review_text_raw"]),
                        "app_b": sample.iloc[j]["app_name"],
                        "comment_b": str(sample.iloc[j]["review_text_raw"]),
                    }
                )
    if not pairs:
        return pd.DataFrame()
    return pd.DataFrame(pairs).sort_values("similarity", ascending=False).head(top_k).reset_index(drop=True)


def aspect_comment_detail_table(long_df: pd.DataFrame, aspect: str, sentiment: str = "Semua") -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    frame = long_df[long_df["aspect"] == aspect].copy()
    if sentiment != "Semua":
        frame = frame[frame["pred_label"] == sentiment]

    if frame.empty:
        return pd.DataFrame()

    out = frame.sort_values("review_date", ascending=False)[
        ["app_name", "review_date", "pred_label", "review_text_raw"]
    ].copy()
    out.rename(
        columns={
            "app_name": "App",
            "review_date": "Tanggal",
            "pred_label": "Sentimen",
            "review_text_raw": "Komentar",
        },
        inplace=True,
    )
    out["Tanggal"] = pd.to_datetime(out["Tanggal"], errors="coerce").dt.date
    return out.reset_index(drop=True)


def sentiment_word_frequencies(long_df: pd.DataFrame, sentiment: str, top_n: int = 120) -> list[tuple[str, int]]:
    if long_df.empty:
        return []

    if sentiment == "Semua":
        source = long_df.drop_duplicates(subset=["review_id_ext"])
    else:
        source = long_df[long_df["pred_label"] == sentiment].drop_duplicates(subset=["review_id_ext"])

    words: list[str] = []
    for text in source["review_text_clean"].fillna("").astype(str):
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
        words.extend([token for token in tokens if token not in STOPWORDS])

    if not words:
        return []
    return Counter(words).most_common(top_n)


def render_sentiment_wordcloud(long_df: pd.DataFrame, sentiment: str) -> None:
    freq_pairs = sentiment_word_frequencies(long_df, sentiment=sentiment, top_n=120)
    if not freq_pairs:
        st.info(f"Belum cukup data kata untuk sentimen {sentiment.lower()}.")
        return

    import importlib.util

    use_wordcloud = importlib.util.find_spec("wordcloud") is not None
    if use_wordcloud:
        from PIL import Image
        from wordcloud import WordCloud

        frequencies = dict(freq_pairs)
        cloud = WordCloud(
            width=900,
            height=420,
            background_color="white",
            colormap="viridis",
            max_words=120,
            collocations=False,
        ).generate_from_frequencies(frequencies)

        image = cloud.to_image()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        st.image(Image.open(buffer), use_container_width=True)
        return

    freq_df = pd.DataFrame(freq_pairs[:30], columns=["word", "count"])
    fig = px.bar(
        freq_df.sort_values("count", ascending=True),
        x="count",
        y="word",
        orientation="h",
        title=f"Top kata sentimen {sentiment.lower()} (fallback tanpa wordcloud)",
    )
    fig.update_traces(marker_color="#4f6f8f")
    st.plotly_chart(chart_theme(fig), use_container_width=True)


def benchmark_note(registry_df: pd.DataFrame, model_id: str) -> str:
    if registry_df.empty or model_id not in registry_df["model_id"].values:
        return "Benchmark model tidak tersedia."

    row = registry_df.loc[registry_df["model_id"] == model_id].iloc[0]
    gold_rank = int(row["rank_gold_subset"]) if pd.notna(row.get("rank_gold_subset")) else None
    weak_rank = int(row["rank_weak_label"]) if pd.notna(row.get("rank_weak_label")) else None
    gold_f1 = float(row["gold_f1_macro"]) if pd.notna(row.get("gold_f1_macro")) else None
    weak_f1 = float(row["weak_f1_macro"]) if pd.notna(row.get("weak_f1_macro")) else None

    parts: list[str] = []
    if gold_rank is not None and gold_f1 is not None:
        parts.append(f"rank gold #{gold_rank} (F1 {gold_f1:.3f})")
    if weak_rank is not None and weak_f1 is not None:
        parts.append(f"rank weak-label #{weak_rank} (F1 {weak_f1:.3f})")
    return " | ".join(parts) if parts else "Belum ada benchmark gold/weak untuk model ini."


def trim_text(text: str, max_chars: int = 220) -> str:
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def extract_salient_phrases(text_series: pd.Series, top_n: int = 3) -> list[str]:
    phrases: list[str] = []
    for text in text_series.fillna("").astype(str):
        tokens = [token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower()) if token not in STOPWORDS]
        bigrams = [" ".join(tokens[i : i + 2]) for i in range(max(len(tokens) - 1, 0))]
        phrases.extend([phrase for phrase in bigrams if len(phrase.split()) == 2])
    if not phrases:
        return []
    return [phrase for phrase, _ in Counter(phrases).most_common(top_n)]


def annotate_aspect_presence(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return long_df.copy()

    base = (
        long_df[
            ["review_id_ext", "source_job_id", "review_text_clean", "review_text_raw"]
        ]
        .drop_duplicates(subset=["review_id_ext", "source_job_id"])
        .copy()
    )
    rows: list[dict[str, object]] = []
    for row in base.itertuples(index=False):
        text = row.review_text_clean if pd.notna(row.review_text_clean) and str(row.review_text_clean).strip() else row.review_text_raw
        details = aspect_presence_details(text)
        payload: dict[str, object] = {
            "review_id_ext": row.review_id_ext,
            "source_job_id": row.source_job_id,
        }
        for aspect, meta in details.items():
            payload[f"{aspect}_present_rule"] = bool(meta["present"])
            hits = list(meta["hits"])
            payload[f"{aspect}_presence_hits"] = int(len(hits))
            payload[f"{aspect}_presence_keywords"] = ", ".join(hits) if hits else "-"
        rows.append(payload)

    presence_df = pd.DataFrame(rows)
    merged = long_df.merge(presence_df, on=["review_id_ext", "source_job_id"], how="left")
    merged["aspect_present_rule"] = False
    merged["aspect_presence_hits"] = 0
    merged["aspect_presence_keywords"] = "-"
    for aspect in ["risk", "trust", "service"]:
        mask = merged["aspect"] == aspect
        merged.loc[mask, "aspect_present_rule"] = merged.loc[mask, f"{aspect}_present_rule"].fillna(False).astype(bool)
        merged.loc[mask, "aspect_presence_hits"] = pd.to_numeric(
            merged.loc[mask, f"{aspect}_presence_hits"], errors="coerce"
        ).fillna(0).astype(int)
        merged.loc[mask, "aspect_presence_keywords"] = (
            merged.loc[mask, f"{aspect}_presence_keywords"].fillna("-").astype(str)
        )
    return merged


def filter_present_aspects(long_df: pd.DataFrame) -> pd.DataFrame:
    annotated = annotate_aspect_presence(long_df)
    if annotated.empty:
        return annotated
    return annotated[annotated["aspect_present_rule"]].reset_index(drop=True)


def aspect_presence_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    annotated = annotate_aspect_presence(long_df)
    if annotated.empty:
        return pd.DataFrame(columns=["aspect", "present_rows", "all_rows", "present_rate_pct"])

    rows: list[dict[str, object]] = []
    for aspect in ["risk", "trust", "service"]:
        subset = annotated[annotated["aspect"] == aspect].copy()
        total = int(len(subset))
        present = int(subset["aspect_present_rule"].sum()) if not subset.empty else 0
        rows.append(
            {
                "aspect": aspect,
                "present_rows": present,
                "all_rows": total,
                "present_rate_pct": round((present / total * 100.0), 1) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def issue_breakdown(long_df: pd.DataFrame, aspect: str) -> pd.DataFrame:
    subset = long_df[(long_df["aspect"] == aspect) & (long_df["pred_label"] == "Negative")].copy()
    if subset.empty:
        return pd.DataFrame(columns=["issue", "count", "share", "dominant_app", "keywords", "sample_phrases"])

    subset["issue"] = GENERAL_ISSUE_LABEL
    subset["matched_keywords"] = [[] for _ in range(len(subset))]
    for idx, row in subset.iterrows():
        label, hits = assign_issue_label(row.get("review_text_clean", row.get("review_text_raw", "")), aspect)
        subset.at[idx, "issue"] = label
        subset.at[idx, "matched_keywords"] = hits

    rows: list[dict[str, object]] = []
    total = len(subset)
    for issue, issue_df in subset.groupby("issue"):
        dominant_app = issue_df["app_name"].value_counts().idxmax() if "app_name" in issue_df.columns and not issue_df["app_name"].dropna().empty else "-"
        keyword_counts = Counter(keyword for hits in issue_df["matched_keywords"] for keyword in hits)
        keywords = ", ".join(keyword for keyword, _ in keyword_counts.most_common(3))
        sample_phrases = ", ".join(extract_salient_phrases(issue_df["review_text_clean"], top_n=2))
        rows.append(
            {
                "issue": issue,
                "count": int(len(issue_df)),
                "share": round(len(issue_df) / total * 100.0, 1),
                "dominant_app": dominant_app,
                "keywords": keywords or "-",
                "sample_phrases": sample_phrases or "-",
            }
        )
    return pd.DataFrame(rows).sort_values(["count", "issue"], ascending=[False, True]).reset_index(drop=True)


def issue_specific_coverage(issue_df: pd.DataFrame) -> str:
    if issue_df.empty:
        return "Belum ada cukup keluhan untuk dibaca."
    total = float(issue_df["count"].sum())
    specific = float(issue_df.loc[issue_df["issue"] != GENERAL_ISSUE_LABEL, "count"].sum())
    share = (specific / total * 100.0) if total else 0.0
    return f"Tema spesifik menjelaskan {share:.0f}%; sisanya masih umum."


def pill_html(text: str, tone: str = "neutral") -> str:
    return f'<span class="pill pill-{tone}">{html.escape(str(text))}</span>'


def scope_strip_html(
    chosen_job: pd.Series,
    model_label: str,
    current_view: str,
    final_reviews: int,
    benchmark_text: str,
    lens_note: str,
) -> str:
    cells = [
        ("Scope", chosen_job["app_name"]),
        ("Model aktif", model_label),
        ("View", current_view),
        ("Final ulasan", f"{final_reviews:,}"),
        ("Periode", f'{chosen_job["date_from"]} s/d {chosen_job["date_to"]}'),
    ]
    cell_html = "".join(
        f'<div class="scope-cell"><div class="scope-label">{html.escape(label)}</div><div class="scope-value">{html.escape(str(value))}</div></div>'
        for label, value in cells
    )
    subline = (
        f'job <code>{html.escape(str(chosen_job["job_id"]))}</code> | '
        f'{html.escape(lens_note)} | {html.escape(benchmark_text)}'
    )
    return (
        f'<div class="scope-strip">'
        f'<div class="eyebrow-mini">Active scope</div>'
        f'<div class="scope-grid">{cell_html}</div>'
        f'<div class="scope-subline">{subline}</div>'
        f'</div>'
    )


def health_row_html(label: str, value: int, tone: str) -> str:
    return (
        f'<div class="health-row">'
        f'<div class="health-row-label">{html.escape(label)}</div>'
        f'<div class="health-track"><div class="health-fill-{tone}" style="width:{max(min(int(value), 100), 0)}%"></div></div>'
        f'<div class="health-row-value">{int(value)}%</div>'
        f'</div>'
    )


def sentiment_tone(label: object, fallback: str = "neutral") -> str:
    mapping = {
        "Positive": "positive",
        "positive": "positive",
        "Neutral": "neutral",
        "neutral": "neutral",
        "Negative": "negative",
        "negative": "negative",
    }
    return mapping.get(str(label), fallback)


def sentiment_distribution_bar_html(rec: pd.Series, show_legend: bool = True) -> str:
    parts = [
        ("positive", int(rec["positive_share"])),
        ("neutral", int(rec["neutral_share"])),
        ("negative", int(rec["negative_share"])),
    ]
    segments = "".join(
        f'<div class="sentiment-segment sentiment-segment-{tone}" style="width:{max(value, 0)}%"></div>'
        for tone, value in parts
    )
    legend = "".join(
        (
            f'<div class="sentiment-legend-row">'
            f'<div class="sentiment-legend-label">{label}</div>'
            f'<div class="sentiment-legend-value">{value}%</div>'
            f'</div>'
        )
        for label, value in [
            ("Positif", int(rec["positive_share"])),
            ("Netral", int(rec["neutral_share"])),
            ("Negatif", int(rec["negative_share"])),
        ]
    )
    if not show_legend:
        return f'<div class="sentiment-stack">{segments}</div>'
    return f'<div class="sentiment-stack">{segments}</div><div class="sentiment-legend">{legend}</div>'


def aspect_health_card_html(rec: pd.Series, aspect: str) -> str:
    dominant_badge = pill_html(
        f"{rec['dominant_sentiment']} {int(rec['dominant_share'])}%",
        sentiment_tone(rec["dominant_sentiment"]),
    )
    bars = sentiment_distribution_bar_html(rec)
    return (
        f'<div class="health-card">'
        f'<div class="health-head">'
        f'<div><div class="health-name">{html.escape(str(rec["aspect"]).title())}</div>'
        f'<div class="health-balance">{html.escape(str(rec["balance_note"]))}</div></div>'
        f'<div class="health-meta">{dominant_badge}</div>'
        f'</div>'
        f'<div class="health-bars">{bars}</div>'
        f'<div class="health-footer">{pill_html("Distribusi sentimen per aspek", "neutral")}</div>'
        f'</div>'
    )


def summary_card_html(label: str, value: str) -> str:
    return (
        f'<div class="summary-card">'
        f'<div class="summary-title">{html.escape(label)}</div>'
        f'<div class="summary-value">{html.escape(value)}</div>'
        f'</div>'
    )


def diagnosis_implication_text(item: pd.Series, aspect: str) -> str:
    issue = str(item.get("issue", "")).strip().lower()
    if issue.startswith("campuran"):
        if aspect == "service":
            return "Keluhan pada Service belum terkonsentrasi pada satu isu utama."
        if aspect == "trust":
            return "Keluhan pada Trust masih tersebar di beberapa isu."
        return "Keluhan pada aspek ini masih tersebar di beberapa isu."

    issue_label = re.sub(r"\s*\(\d+%\)\s*$", "", str(item.get("issue", "")).strip())
    if issue_label:
        return f"Keluhan paling banyak terkait {issue_label.lower()}."
    return "Aspek ini masih perlu dibaca hati-hati."


def diagnosis_distribution_html(value: object) -> str:
    text = str(value or "").strip()
    if not text or text == "-":
        return '<div class="diagnosis-row-value">-</div>'

    parts = [part.strip() for part in text.splitlines() if part.strip()]
    if len(parts) <= 1:
        return f'<div class="diagnosis-row-value diagnosis-row-value-text">{html.escape(text)}</div>'

    rendered_lines: list[str] = []
    for part in parts:
        lower = part.lower()
        tone_class = ""
        if lower.startswith("negatif:"):
            tone_class = " diagnosis-dist-line-negative"
        elif lower.startswith("positif:"):
            tone_class = " diagnosis-dist-line-positive"
        rendered_lines.append(
            f'<div class="diagnosis-dist-line{tone_class}">{html.escape(part)}</div>'
        )
    lines = "".join(rendered_lines)
    return f'<div class="diagnosis-distribution">{lines}</div>'


def diagnosis_summary_html(item: pd.Series, aspect: str) -> str:
    rows_html = (
        f'<div class="diagnosis-row">'
        f'<div class="diagnosis-row-label">Fokus isu</div>'
        f'<div class="diagnosis-row-value diagnosis-row-value-text">{html.escape(str(item.get("issue", "-")))}</div>'
        f'</div>'
        f'<div class="diagnosis-row">'
        f'<div class="diagnosis-row-label">Arah tren</div>'
        f'<div class="diagnosis-row-value diagnosis-row-value-text">{html.escape(str(item.get("trend", "-")))}</div>'
        f'</div>'
        f'<div class="diagnosis-row">'
        f'<div class="diagnosis-row-label">Distribusi sinyal</div>'
        f'{diagnosis_distribution_html(item.get("worst_app", "-"))}'
        f'</div>'
    )
    return f"""
    <div class="diagnosis-summary">
        <div class="eyebrow-mini">So what</div>
        <div class="card-title">{html.escape(str(item['title']))}</div>
        <div class="diagnosis-lead">{html.escape(diagnosis_implication_text(item, aspect))}</div>
        <div class="diagnosis-rows">
            {rows_html}
        </div>
    </div>
    """


def issue_map_card_html(title: str, issue_df: pd.DataFrame, coverage_text: str, aspect: str) -> str:
    rows_html = []
    for _, issue_row in issue_df.iterrows():
        keyword_text = html.escape(str(issue_row["keywords"]))
        sample_text = html.escape(str(issue_row["sample_phrases"]))
        if str(issue_row["issue"]) == GENERAL_ISSUE_LABEL:
            submeta_html = "pola masih umum; cek review bukti langsung"
        else:
            submeta_parts = []
            if keyword_text != "-":
                submeta_parts.append(f'kata: {keyword_text}')
            if sample_text != "-":
                submeta_parts.append(f'frasa: {sample_text}')
            submeta_html = " | ".join(submeta_parts) if submeta_parts else "pola masih umum"
        rows_html.append(
            (
                f'<div class="issue-row">'
                f'<div class="issue-title">{pill_html(issue_row["issue"], aspect)}</div>'
                f'<div class="issue-meta"><strong>{issue_row["share"]:.0f}%</strong> keluhan negatif • dominan di '
                f'<strong>{html.escape(str(issue_row["dominant_app"]))}</strong></div>'
                f'<div class="issue-submeta">{submeta_html}</div>'
                f'</div>'
            )
        )
    return (
        f'<div class="issue-card">'
        f'<div class="eyebrow-mini">Interpretive layer</div>'
        f'<div class="card-title">{html.escape(title)}</div>'
        f'<div class="trust-note">{html.escape(coverage_text)}</div>'
        f'{"".join(rows_html)}'
        f'</div>'
    )


def aspect_trend_signal(long_df: pd.DataFrame, aspect: str) -> str:
    subset = long_df[long_df["aspect"] == aspect].copy()
    if subset.empty:
        return "Trend belum tersedia."

    daily = (
        subset.assign(day=subset["review_date"].dt.date, is_negative=subset["pred_label"].eq("Negative").astype(int))
        .groupby("day", as_index=False)
        .agg(neg_share=("is_negative", "mean"), volume=("review_id_ext", "nunique"))
        .sort_values("day")
    )
    if len(daily) < 6:
        return "Trend belum cukup panjang."

    recent = daily.tail(3)["neg_share"].mean()
    prev = daily.tail(6).head(3)["neg_share"].mean()
    delta = (recent - prev) * 100.0
    if delta >= 5:
        return "Cenderung naik dibanding 3 hari sebelumnya."
    if delta <= -5:
        return "Cenderung turun dibanding 3 hari sebelumnya."
    return "Relatif stabil vs 3 hari."


def worst_app_for_aspect(long_df: pd.DataFrame, aspect: str) -> str:
    subset = long_df[long_df["aspect"] == aspect].copy()
    if subset.empty or "app_name" not in subset.columns:
        return "-"

    app_scores = (
        subset.assign(
            is_negative=subset["pred_label"].eq("Negative").astype(int),
            is_positive=subset["pred_label"].eq("Positive").astype(int),
        )
        .groupby("app_name", as_index=False)
        .agg(
            negative_share=("is_negative", "mean"),
            positive_share=("is_positive", "mean"),
            volume=("review_id_ext", "nunique"),
        )
    )
    if app_scores.empty:
        return "-"
    app_scores["negative_pct"] = (app_scores["negative_share"] * 100).round(0).astype(int)
    app_scores["positive_pct"] = (app_scores["positive_share"] * 100).round(0).astype(int)

    if len(app_scores) == 1:
        row = app_scores.iloc[0]
        return (
            f"Negatif: {row['app_name']} {int(row['negative_pct'])}%\n"
            f"Positif: {row['app_name']} {int(row['positive_pct'])}%"
        )

    preferred_order = ["Kredivo", "Akulaku"]
    ordered_rows: list[pd.Series] = []
    for app_name in preferred_order:
        match = app_scores[app_scores["app_name"].astype(str) == app_name]
        if not match.empty:
            ordered_rows.append(match.iloc[0])
    for _, row in app_scores.iterrows():
        if str(row["app_name"]) not in preferred_order:
            ordered_rows.append(row)

    negative_line = "Negatif: " + " | ".join(
        f"{row['app_name']} {int(row['negative_pct'])}%"
        for row in ordered_rows
    )
    positive_line = "Positif: " + " | ".join(
        f"{row['app_name']} {int(row['positive_pct'])}%"
        for row in ordered_rows
    )
    return f"{negative_line}\n{positive_line}"


def top_issue_for_aspect(long_df: pd.DataFrame, aspect: str) -> str:
    issues = issue_breakdown(long_df, aspect)
    if issues.empty:
        return "Belum ada pola kuat"
    top = issues.iloc[0]
    if top["issue"] == GENERAL_ISSUE_LABEL:
        specific = issues[issues["issue"] != GENERAL_ISSUE_LABEL]
        if not specific.empty:
            next_issue = specific.iloc[0]
            return f"Campuran; sinyal {next_issue['issue']} ({next_issue['share']:.0f}%)"
        return "Campuran; belum cukup spesifik"
    return f"{top['issue']} ({top['share']:.0f}%)"


def aspect_diagnosis_table(
    long_df: pd.DataFrame,
    score_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for aspect in ["risk", "trust", "service"]:
        row = score_df[score_df["aspect"] == aspect]
        if row.empty:
            continue
        rec = row.iloc[0]
        aspect_volume = int(
            long_df.loc[long_df["aspect"] == aspect, "review_id_ext"].nunique()
        ) if not long_df.empty else 0
        rows.append(
            {
                "aspect": aspect,
                "title": ASPECT_LABEL_MAP[aspect],
                "dominant_sentiment": rec["dominant_sentiment"],
                "dominant_share": int(rec["dominant_share"]),
                "negative_share": int(rec["negative_share"]),
                "positive_share": int(rec["positive_share"]),
                "neutral_share": int(rec["neutral_share"]),
                "balance_note": rec["balance_note"],
                "volume": aspect_volume,
                "issue": top_issue_for_aspect(long_df, aspect),
                "worst_app": worst_app_for_aspect(long_df, aspect),
                "trend": aspect_trend_signal(long_df, aspect),
            }
        )
    return pd.DataFrame(rows)


def diagnosis_examples_frame(
    long_df: pd.DataFrame,
    aspect: str,
    sentiment: str,
    limit: int,
) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    frame = long_df[
        (long_df["aspect"] == aspect) & (long_df["pred_label"] == sentiment)
    ].copy()
    if frame.empty:
        return frame

    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.0)
    frame["review_date"] = pd.to_datetime(frame["review_date"], errors="coerce")
    frame["review_text_raw"] = frame["review_text_raw"].fillna("").astype(str)
    frame["review_text_clean"] = frame["review_text_clean"].fillna(frame["review_text_raw"]).astype(str)
    frame["text_len"] = frame["review_text_raw"].str.len()
    frame = frame[frame["review_text_raw"].str.strip().str.len() > 0].copy()
    if frame.empty:
        return frame

    frame["sample_score"] = (
        frame["confidence"] * 3.0
        + frame["text_len"].clip(lower=40, upper=240) / 240.0
        + frame.get("aspect_presence_hits", 0)
    )
    ranked = frame.sort_values(
        ["sample_score", "confidence", "review_date", "text_len"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    ranked = ranked.drop_duplicates(subset=["review_id_ext"], keep="first")
    return ranked.head(limit).reset_index(drop=True)


def diagnosis_example_html(row: pd.Series, sentiment: str) -> str:
    date_value = pd.to_datetime(row.get("review_date"), errors="coerce")
    date_text = str(date_value.date()) if pd.notna(date_value) else "-"
    meta = " • ".join(
        [
            html.escape(str(row.get("app_name", "-"))),
            html.escape(date_text),
            f"conf {float(row.get('confidence', 0.0)):.2f}",
        ]
    )
    return f"""
    <div class="example-card example-card-{sentiment_tone(sentiment)}">
        <div class="example-meta">
            {meta}<br/>{pill_html(sentiment, sentiment_tone(sentiment))}
        </div>
        <div class="example-quote">"{html.escape(trim_text(str(row.get("review_text_raw", "")), max_chars=240))}"</div>
    </div>
    """


def render_example_native_card(row: pd.Series, sentiment: str) -> None:
    date_value = pd.to_datetime(row.get("review_date"), errors="coerce")
    date_text = str(date_value.date()) if pd.notna(date_value) else "-"
    meta = " | ".join(
        [
            str(row.get("app_name", "-")),
            date_text,
            f"conf {float(row.get('confidence', 0.0)):.2f}",
        ]
    )
    st.markdown(
        f'<div class="example-native-card example-native-card-{sentiment_tone(sentiment)}">'
        f'<div class="example-native-meta">{html.escape(meta)}</div>'
        f'<div class="example-native-text">{html.escape(trim_text(str(row.get("review_text_raw", "")), max_chars=220))}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_diagnosis_examples(long_df: pd.DataFrame, aspect: str) -> None:
    example_plan = [("Positive", 4), ("Neutral", 2), ("Negative", 4)]
    for idx, (sentiment, limit) in enumerate(example_plan):
        if idx > 0:
            st.markdown('<div class="section-spacer-md"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="example-section-title">{html.escape(sentiment)} ({limit})</div>',
            unsafe_allow_html=True,
        )
        sample_df = diagnosis_examples_frame(long_df, aspect, sentiment, limit)
        if sample_df.empty:
            st.info(f"Belum ada contoh {sentiment.lower()} untuk aspek ini.")
            continue
        st.markdown(
            f'<div class="example-section-note">Menampilkan {len(sample_df)} contoh review yang paling representatif untuk sentimen ini.</div>',
            unsafe_allow_html=True,
        )
        for _, row in sample_df.iterrows():
            render_example_native_card(row, sentiment)
        if len(sample_df) < limit:
            st.caption(
                f"Contoh yang tersedia saat ini {len(sample_df)} dari target {limit}."
            )


def conclusion_card_html(title: str, body: str, tone: str, variant: str = "default") -> str:
    return (
        f'<div class="conclusion-card conclusion-card-{tone} conclusion-card-{variant}">'
        f'<div class="conclusion-title">{html.escape(title)}</div>'
        f'<div class="conclusion-copy">{html.escape(body)}</div>'
        f'</div>'
    )


def split_summary_sentences(text: object, limit: int = 2) -> list[str]:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if not sentences:
        return [cleaned]
    return sentences[:limit]


def summary_metric_chips(metrics: dict[str, object] | None) -> list[str]:
    if not isinstance(metrics, dict):
        return []

    sentiment_map = {
        "Positive": "Positif",
        "Neutral": "Netral",
        "Negative": "Negatif",
    }
    chips: list[str] = []
    dominant = metrics.get("dominant_sentiment")
    dominant_share = metrics.get("dominant_share")
    if dominant is not None and dominant_share is not None:
        try:
            chips.append(f"Dominan: {sentiment_map.get(str(dominant), str(dominant))} {float(dominant_share):.1f}%")
        except (TypeError, ValueError):
            chips.append(f"Dominan: {sentiment_map.get(str(dominant), str(dominant))}")

    best_aspect = metrics.get("best_aspect")
    if best_aspect:
        chips.append(f"Kuat: {ASPECT_LABEL_MAP.get(str(best_aspect), str(best_aspect).title())}")

    worst_aspect = metrics.get("worst_aspect")
    if worst_aspect:
        chips.append(f"Rawan: {ASPECT_LABEL_MAP.get(str(worst_aspect), str(worst_aspect).title())}")

    issue = metrics.get("issue")
    if issue:
        chips.append(f"Isu: {trim_text(str(issue), max_chars=42)}")

    trend = metrics.get("trend")
    if trend:
        chips.append(trim_text(str(trend), max_chars=44))

    negative_share = metrics.get("negative_share")
    if negative_share is not None and "aspect" in metrics:
        try:
            chips.append(f"Negatif: {float(negative_share):.1f}%")
        except (TypeError, ValueError):
            pass

    app_name = metrics.get("app_name")
    if app_name:
        chips.append(f"App: {app_name}")

    seen: set[str] = set()
    out: list[str] = []
    for chip in chips:
        text = re.sub(r"\s+", " ", str(chip)).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out[:4]


def render_compact_summary_card(card: object, default_title: str, default_tone: str, variant: str = "default") -> str:
    if isinstance(card, str) and card.strip():
        return card

    if not isinstance(card, dict):
        return conclusion_card_html(default_title, "Belum ada data yang cukup untuk membentuk kesimpulan.", default_tone, variant)

    title = str(card.get("title", default_title))
    tone = sentiment_tone(card.get("tone", default_tone), default_tone)
    body = str(card.get("body", card.get("text", ""))).strip()
    sentences = split_summary_sentences(body, limit=2)
    lead = sentences[0] if sentences else "Belum ada data yang cukup untuk membentuk kesimpulan."
    subline = sentences[1] if len(sentences) > 1 else ""

    evidence_items: list[str] = summary_metric_chips(card.get("metrics"))

    seen: set[str] = set()
    compact_evidence: list[str] = []
    for item in evidence_items:
        text = re.sub(r"\s+", " ", str(item)).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        compact_evidence.append(text)
    compact_evidence = compact_evidence[:3]

    evidence_html = ""
    if compact_evidence:
        chips_html = "".join(f'<span class="conclusion-chip">{html.escape(item)}</span>' for item in compact_evidence)
        evidence_html = f'<div class="conclusion-evidence-row">{chips_html}</div>'

    subline_html = f'<div class="conclusion-subline">{html.escape(subline)}</div>' if subline else ""
    return (
        f'<div class="conclusion-card conclusion-card-{tone} conclusion-card-{variant}">'
        f'<div class="conclusion-title">{html.escape(title)}</div>'
        f'<div class="conclusion-lead">{html.escape(lead)}</div>'
        f'{subline_html}'
        f'{evidence_html}'
        f'</div>'
    )


def normalize_summary_conclusion_payload(payload: object) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None

    def coerce_card(value: object, default_title: str, default_tone: str, variant: str = "default") -> str:
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            return render_compact_summary_card(value, default_title, default_tone, variant)
        return conclusion_card_html(default_title, "Belum ada data yang cukup untuk membentuk kesimpulan.", default_tone, variant)

    overall = payload.get("overall") or payload.get("overview") or payload.get("gambaran")
    signal = payload.get("signal") or payload.get("focus") or payload.get("action") or payload.get("insight")
    meaning = payload.get("meaning") or payload.get("final_meaning") or payload.get("closing")
    apps = payload.get("apps") or payload.get("app_cards") or payload.get("per_app") or []

    if overall is None or signal is None:
        return None

    normalized_apps: list[str] = []
    if isinstance(apps, list):
        for idx, item in enumerate(apps):
            default_title = "Kredivo" if idx == 0 else "Akulaku" if idx == 1 else f"App {idx + 1}"
            normalized_apps.append(coerce_card(item, default_title, "neutral", "app"))

    return {
        "overall": coerce_card(overall, "Gambaran Pengalaman", "neutral", "lead"),
        "signal": coerce_card(signal, "Sinyal yang Perlu Diperhatikan", "neutral", "focus"),
        "meaning": coerce_card(meaning, "Makna Akhir", "neutral", "close") if meaning is not None else "",
        "apps": normalized_apps,
    }


def build_summary_conclusion_payload(long_df: pd.DataFrame, score_df: pd.DataFrame) -> dict[str, object]:
    if dashboard_summary_payload is not None:
        try:
            payload = dashboard_summary_payload(long_df=long_df, score_df=score_df)
        except TypeError:
            payload = dashboard_summary_payload(long_df, score_df)
        normalized = normalize_summary_conclusion_payload(payload)
        if normalized is not None:
            return normalized
    fallback_payload = executive_conclusion_payload(long_df, score_df)
    return normalize_summary_conclusion_payload(fallback_payload) or fallback_payload


def app_conclusion_card(long_df: pd.DataFrame, app_name: str) -> str:
    app_df = long_df[long_df["app_name"] == app_name].copy()
    if app_df.empty:
        return conclusion_card_html(
            app_name,
            "Belum ada data cukup untuk membentuk ringkasan app ini.",
            "neutral",
            "app",
        )

    score_df = aspect_score_table(app_df)
    if score_df.empty:
        return conclusion_card_html(
            app_name,
            "Distribusi sentimen app ini belum cukup lengkap.",
            "neutral",
            "app",
        )

    dominant = app_df["pred_label"].value_counts(normalize=True).mul(100).round(1).to_dict()
    dominant_label = max(dominant, key=dominant.get)
    negative_row = score_df.sort_values(["negative_share", "positive_share"], ascending=[False, False]).iloc[0]
    positive_row = score_df.sort_values(["positive_share", "neutral_share"], ascending=[False, False]).iloc[0]

    body = (
        f"Iklim sentimen di {app_name} saat ini didominasi {dominant_label.lower()} "
        f"({dominant.get(dominant_label, 0):.1f}%). Tekanan negatif paling kuat muncul pada aspek "
        f"{ASPECT_LABEL_MAP[str(negative_row['aspect'])]} ({int(negative_row['negative_share'])}%), "
        f"sementara kekuatan terbaik terlihat pada aspek {ASPECT_LABEL_MAP[str(positive_row['aspect'])]} "
        f"({int(positive_row['positive_share'])}%)."
    )
    return conclusion_card_html(app_name, body, sentiment_tone(dominant_label), "app")


def executive_conclusion_payload(
    long_df: pd.DataFrame,
    score_df: pd.DataFrame,
) -> dict[str, object]:
    if long_df.empty or score_df.empty:
        return {
            "overall": conclusion_card_html(
                "Ringkasan Umum",
                "Belum ada data yang cukup untuk membentuk kesimpulan.",
                "neutral",
                "lead",
            ),
            "focus": "",
            "apps": [],
        }

    total_sentiment = (
        long_df["pred_label"].value_counts(normalize=True).mul(100).round(1).to_dict()
    )
    dominant_sentiment = max(total_sentiment, key=total_sentiment.get)

    positive_row = score_df.sort_values(
        ["positive_share", "neutral_share"], ascending=[False, False]
    ).iloc[0]
    negative_row = score_df.sort_values(
        ["negative_share", "positive_share"], ascending=[False, False]
    ).iloc[0]
    positive_aspect = str(positive_row["aspect"])
    negative_aspect = str(negative_row["aspect"])

    overall = conclusion_card_html(
        "Gambaran Pengalaman",
        (
            f"Secara keseluruhan, pengalaman pakai masih didominasi {dominant_sentiment.lower()}. "
            f"Hal yang paling sering muncul sebagai nilai positif ada di aspek {ASPECT_LABEL_MAP[positive_aspect]}, "
            f"sementara tekanan paling kuat ada di aspek {ASPECT_LABEL_MAP[negative_aspect]}."
        ),
        sentiment_tone(dominant_sentiment),
        "lead",
    )
    signal = conclusion_card_html(
        "Sinyal yang Perlu Diperhatikan",
        (
            f"Perhatian utama ada pada aspek {ASPECT_LABEL_MAP[negative_aspect]} karena area ini paling sering memicu keluhan. "
            f"Di sisi lain, kekuatan pada aspek {ASPECT_LABEL_MAP[positive_aspect]} tetap perlu dijaga."
        ),
        "negative",
        "focus",
    )
    meaning = conclusion_card_html(
        "Makna Akhir",
        (
            f"Kalau pola ini terus berulang, pengalaman pakai bisa terasa kurang stabil walaupun masih ada sisi yang membantu. "
            f"Artinya, area {ASPECT_LABEL_MAP[negative_aspect]} perlu dibenahi sambil menjaga kekuatan di {ASPECT_LABEL_MAP[positive_aspect]}."
        ),
        "neutral",
        "close",
    )
    app_cards: list[str] = []
    raw_app_names = [str(name) for name in long_df["app_name"].dropna().astype(str).unique().tolist()]
    ordered_app_names = [name for name in ["Kredivo", "Akulaku"] if name in raw_app_names]
    ordered_app_names.extend(sorted(name for name in raw_app_names if name not in ordered_app_names))
    for app_name in ordered_app_names:
        app_cards.append(app_conclusion_card(long_df, app_name))
    return {
        "overall": overall,
        "signal": signal,
        "meaning": meaning,
        "apps": app_cards,
    }


def review_card_payload(row: pd.Series, title: str, rationale: str) -> dict[str, str]:
    issue_label = coalesce_text(row.get("issue"), GENERAL_ISSUE_LABEL)
    issue_keywords = coalesce_text(row.get("issue_keywords"), "-")
    presence_keywords = coalesce_text(row.get("aspect_presence_keywords"), "-")
    presence_hits_raw = pd.to_numeric(row.get("aspect_presence_hits"), errors="coerce")
    presence_hits = int(presence_hits_raw) if pd.notna(presence_hits_raw) else 0
    signal_parts = [f"Issue: {issue_label}"]
    if issue_keywords != "-":
        signal_parts.append(f"Kata kunci: {issue_keywords}")
    elif presence_keywords != "-":
        signal_parts.append(f"Sinyal aspek: {presence_keywords}")
    signal_parts.append(f"Presence hits: {presence_hits}")
    return {
        "title": title,
        "app": str(row.get("app_name", "-")),
        "date": str(pd.to_datetime(row.get("review_date"), errors="coerce").date()) if pd.notna(pd.to_datetime(row.get("review_date"), errors="coerce")) else "-",
        "sentiment": str(row.get("pred_label", "-")),
        "confidence": f"{float(row.get('confidence', 0.0)):.2f}",
        "quote": trim_text(str(row.get("review_text_raw", ""))),
        "issue": issue_label,
        "signal": " | ".join(signal_parts),
        "rationale": rationale,
    }


def evidence_card_html(card: dict[str, str], aspect: str) -> str:
    issue_tone = sentiment_tone(card.get("sentiment"))
    return f"""
    <div class="evidence-card evidence-card-{html.escape(aspect)}">
        <div class="eyebrow-mini">Review evidence</div>
        <div class="card-title">{html.escape(card.get("title", "-"))}</div>
        <div class="evidence-meta">
            {html.escape(card.get("app", "-"))} • {html.escape(card.get("date", "-"))} •
            {html.escape(card.get("sentiment", "-"))} • conf {html.escape(card.get("confidence", "-"))}
        </div>
        <div class="evidence-pill-row">
            {pill_html(card.get("issue", GENERAL_ISSUE_LABEL), aspect)}
            {pill_html(card.get("sentiment", "-"), issue_tone)}
        </div>
        <div class="evidence-signal">{html.escape(card.get("signal", "-"))}</div>
        <div class="evidence-rationale">{html.escape(card.get("rationale", "-"))}</div>
        <div class="evidence-quote">"{html.escape(card.get("quote", "-"))}"</div>
    </div>
    """


def worst_app_name_for_evidence(long_df: pd.DataFrame, aspect: str) -> str:
    subset = long_df[long_df["aspect"] == aspect].copy()
    if subset.empty or "app_name" not in subset.columns:
        return "-"

    app_scores = (
        subset.assign(is_negative=subset["pred_label"].eq("Negative").astype(int))
        .groupby("app_name", as_index=False)
        .agg(negative_share=("is_negative", "mean"), volume=("review_id_ext", "nunique"))
    )
    if app_scores.empty:
        return "-"
    app_scores = app_scores.sort_values(["negative_share", "volume"], ascending=[False, False])
    return str(app_scores.iloc[0]["app_name"])


def evidence_primary_issue(long_df: pd.DataFrame, aspect: str) -> str:
    issues = issue_breakdown(long_df, aspect)
    if issues.empty:
        return GENERAL_ISSUE_LABEL
    specific = issues[issues["issue"] != GENERAL_ISSUE_LABEL]
    if not specific.empty:
        return str(specific.iloc[0]["issue"])
    return str(issues.iloc[0]["issue"])


def is_generic_evidence_text(text: object) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    if not normalized:
        return True
    generic_patterns = {
        "parah",
        "jelek",
        "buruk",
        "kecewa",
        "sampah",
        "parah banget",
        "jelek banget",
        "gak bagus",
        "ga bagus",
        "tidak bagus",
    }
    if normalized in generic_patterns:
        return True
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", normalized)
    if len(tokens) < 4:
        return True
    if len(set(tokens)) <= 2 and len(tokens) <= 6:
        return True
    return False


def evidence_candidates(long_df: pd.DataFrame, aspect: str) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    frame = long_df[(long_df["aspect"] == aspect) & (long_df["pred_label"] == "Negative")].copy()
    if frame.empty:
        return frame

    frame["review_text_raw"] = frame["review_text_raw"].fillna("").astype(str)
    frame["review_text_clean"] = frame["review_text_clean"].fillna(frame["review_text_raw"]).astype(str)
    frame["char_len"] = frame["review_text_raw"].str.len()
    frame["token_len"] = frame["review_text_raw"].str.findall(r"[a-zA-Z][a-zA-Z0-9_-]*").str.len()
    frame["parsed_review_date"] = pd.to_datetime(frame["review_date"], errors="coerce")
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.0)
    frame["aspect_presence_hits"] = pd.to_numeric(frame["aspect_presence_hits"], errors="coerce").fillna(0).astype(int)

    issue_labels: list[str] = []
    issue_keywords: list[str] = []
    issue_keyword_hits: list[int] = []
    generic_flags: list[bool] = []
    base_scores: list[float] = []

    for row in frame.itertuples(index=False):
        text = row.review_text_clean if str(row.review_text_clean).strip() else row.review_text_raw
        label, hits = assign_issue_label(text, aspect)
        issue_labels.append(label)
        issue_keywords.append(", ".join(hits) if hits else "-")
        issue_keyword_hits.append(len(hits))
        is_generic = is_generic_evidence_text(row.review_text_raw)
        generic_flags.append(is_generic)

        char_len = int(getattr(row, "char_len"))
        token_len = int(getattr(row, "token_len"))
        length_score = -2 if char_len < 40 or token_len < 6 else 2 if 80 <= char_len <= 320 else 1 if 40 <= char_len <= 500 else 0
        base_score = (
            3 * float(getattr(row, "confidence"))
            + min(int(getattr(row, "aspect_presence_hits")), 3)
            + min(len(hits), 3)
            + (2 if label != GENERAL_ISSUE_LABEL else 0)
            + length_score
            + (-2 if is_generic else 0)
        )
        base_scores.append(round(float(base_score), 3))

    frame["issue"] = issue_labels
    frame["issue_keywords"] = issue_keywords
    frame["issue_keyword_hits"] = issue_keyword_hits
    frame["generic_comment"] = generic_flags
    frame["base_score"] = base_scores
    filtered = frame[~frame["generic_comment"]].copy()
    if filtered.empty:
        filtered = frame.copy()
    return filtered.reset_index(drop=True)


def select_evidence_row(
    candidates: pd.DataFrame,
    score_col: str,
    used_review_ids: set[str],
    used_apps: set[str],
    used_issues: set[str],
) -> pd.Series | None:
    if candidates.empty:
        return None

    available = candidates[~candidates["review_id_ext"].astype(str).isin(used_review_ids)].copy()
    if available.empty:
        return None

    available["selection_score"] = pd.to_numeric(available[score_col], errors="coerce").fillna(0.0)
    available["selection_score"] += (~available["app_name"].astype(str).isin(used_apps)).astype(float) * 0.25
    available["selection_score"] += (~available["issue"].astype(str).isin(used_issues)).astype(float) * 0.25

    ranked = available.sort_values(
        ["selection_score", "confidence", "issue_keyword_hits", "aspect_presence_hits", "char_len", "parsed_review_date"],
        ascending=[False, False, False, False, False, False],
        na_position="last",
    )
    if ranked.empty:
        return None
    return ranked.iloc[0]


def representative_evidence_cards(long_df: pd.DataFrame, aspect: str) -> list[dict[str, str]]:
    if long_df.empty:
        return []

    candidates = evidence_candidates(long_df, aspect)
    if candidates.empty:
        return []

    cards: list[dict[str, str]] = []
    used_review_ids: set[str] = set()
    used_apps: set[str] = set()
    used_issues: set[str] = set()
    top_issue = evidence_primary_issue(long_df, aspect)

    representative_pool = candidates.copy()
    representative_pool["representative_score"] = representative_pool["base_score"] + (
        representative_pool["issue"].eq(top_issue).astype(float) * 2.0
    )
    representative = select_evidence_row(
        representative_pool,
        "representative_score",
        used_review_ids,
        used_apps,
        used_issues,
    )
    if representative is not None:
        cards.append(
            review_card_payload(
                representative,
                "Bukti isu dominan",
                f"Review ini paling menjelaskan isu utama pada aspek ini: {top_issue.lower()}",
            )
        )
        used_review_ids.add(str(representative["review_id_ext"]))
        used_apps.add(str(representative["app_name"]))
        used_issues.add(str(representative["issue"]))

    strongest_pool = candidates.copy()
    strongest_pool["strongest_score"] = strongest_pool["base_score"] + (
        strongest_pool["char_len"].ge(120).astype(float) * 1.0
    )
    strongest = select_evidence_row(
        strongest_pool,
        "strongest_score",
        used_review_ids,
        used_apps,
        used_issues,
    )
    if strongest is not None:
        cards.append(
            review_card_payload(
                strongest,
                "Bukti sinyal terkuat",
                "Dipilih karena issue cukup spesifik, confidence tinggi, dan isi review cukup informatif.",
            )
        )
        used_review_ids.add(str(strongest["review_id_ext"]))
        used_apps.add(str(strongest["app_name"]))
        used_issues.add(str(strongest["issue"]))

    worst_app = worst_app_name_for_evidence(long_df, aspect)
    app_pool = candidates[candidates["app_name"] == worst_app].copy() if worst_app and worst_app != "-" else pd.DataFrame()
    if app_pool.empty:
        app_pool = candidates.copy()
    app_pool["worst_app_score"] = app_pool["base_score"] + (
        app_pool["issue"].eq(top_issue).astype(float) * 1.0
    )
    app_specific = select_evidence_row(
        app_pool,
        "worst_app_score",
        used_review_ids,
        used_apps,
        used_issues,
    )
    if app_specific is not None:
        cards.append(
            review_card_payload(
                app_specific,
                "Bukti dari app terdampak",
                f"App dengan tekanan negatif tertinggi saat ini: {worst_app or '-'}",
            )
        )
    return cards[:3]


def model_compare_frame(registry_df: pd.DataFrame) -> pd.DataFrame:
    if registry_df.empty:
        return pd.DataFrame()

    compare = registry_df[
        [
            "model_id",
            "display_name",
            "gold_f1_macro",
            "gold_accuracy",
            "weak_f1_macro",
            "weak_accuracy",
            "training_time_seconds",
            "rank_gold_subset",
            "rank_weak_label",
        ]
    ].copy()
    numeric_cols = [
        "gold_f1_macro",
        "gold_accuracy",
        "weak_f1_macro",
        "weak_accuracy",
        "training_time_seconds",
        "rank_gold_subset",
        "rank_weak_label",
    ]
    for col in numeric_cols:
        compare[col] = pd.to_numeric(compare[col], errors="coerce")
    return compare.sort_values(["rank_gold_subset", "rank_weak_label", "display_name"], na_position="last")


def top_model_row_by_metric(compare_df: pd.DataFrame, metric: str) -> pd.Series | None:
    if compare_df.empty or metric not in compare_df.columns:
        return None
    ranked = compare_df[pd.notna(compare_df[metric])].sort_values(metric, ascending=False, na_position="last")
    if ranked.empty:
        return None
    return ranked.iloc[0]


def gold_per_aspect_frame(summary: dict) -> pd.DataFrame:
    models = summary.get("models_evaluated", [])
    rows: list[dict[str, object]] = []
    for model in models:
        per_aspect = model.get("sentiment_present_only", {}).get("per_aspect", {})
        for aspect, metrics in per_aspect.items():
            rows.append(
                {
                    "model_id": model.get("model_name"),
                    "aspect": aspect,
                    "accuracy": metrics.get("accuracy"),
                    "f1_macro": metrics.get("f1_macro"),
                }
            )
    return pd.DataFrame(rows)


def gold_failure_pattern_frame(summary: dict) -> pd.DataFrame:
    models = summary.get("models_evaluated", [])
    rows: list[dict[str, object]] = []
    for model in models:
        absent = model.get("aspect_absent_diagnostics", {})
        rows.append(
            {
                "model_id": model.get("model_name"),
                "absent_rows": absent.get("n_rows"),
                "absent_mean_confidence": absent.get("mean_confidence"),
                "pred_negative_when_absent": absent.get("predicted_label_distribution", {}).get("Negative", 0),
            }
        )
    return pd.DataFrame(rows)


def gold_model_summary_lookup(summary: dict, model_id: str) -> dict:
    for model in summary.get("models_evaluated", []):
        if model.get("model_name") == model_id:
            return model
    return {}


def per_aspect_winner_frame(gold_aspect_df: pd.DataFrame, registry_df: pd.DataFrame) -> pd.DataFrame:
    if gold_aspect_df.empty:
        return pd.DataFrame()

    best_rows = []
    for aspect, aspect_df in gold_aspect_df.groupby("aspect"):
        ranked = aspect_df.sort_values("f1_macro", ascending=False)
        top = ranked.iloc[0]
        display_name = registry_df.loc[registry_df["model_id"] == top["model_id"], "display_name"]
        best_rows.append(
            {
                "aspect": aspect,
                "model_id": top["model_id"],
                "model": display_name.iloc[0] if not display_name.empty else top["model_id"],
                "f1_macro": top["f1_macro"],
                "accuracy": top["accuracy"],
            }
        )
    return pd.DataFrame(best_rows).sort_values("aspect").reset_index(drop=True)


def failure_case_frame(model_summary: dict, limit: int = 3) -> pd.DataFrame:
    cases = model_summary.get("aspect_absent_diagnostics", {}).get("top_confident_absent_cases", [])[:limit]
    if not cases:
        return pd.DataFrame()
    rows = []
    for case in cases:
        rows.append(
            {
                "Aspek": case.get("aspect"),
                "Prediksi model": case.get("pred_label"),
                "Confidence": round(float(case.get("pred_confidence", 0.0)), 3),
                "Kenapa tricky": case.get("notes", ""),
                "Review": trim_text(case.get("review_text", ""), max_chars=180),
            }
        )
    return pd.DataFrame(rows)


def render_all_in_one_page(store: DashboardStore, registry_df: pd.DataFrame) -> None:
    if "fetch_notice" in st.session_state:
        st.success(st.session_state.pop("fetch_notice"))
    if "fetch_error" in st.session_state:
        st.error(st.session_state.pop("fetch_error"))

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Page 1 - Executive ABSA View</div>
            <h2 style="margin:0.2rem 0 0.3rem 0;">Pahami kondisi setiap aspek, masalah utama, dan kesimpulan sentimennya</h2>
            <div class="section-intro">
                Halaman ini sengaja non-teknis: fokus pada distribusi sentimen, diagnosis singkat, trend, dan ringkasan akhir yang cepat dibaca.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if registry_df.empty:
        st.error("Model registry kosong. Pastikan model artifacts tersedia agar analisis live bisa dijalankan.")
        return

    default_row = default_model_row(registry_df)
    default_model_id = default_row["model_id"] if default_row is not None else registry_df.iloc[0]["model_id"]

    with st.expander("Refresh scope atau jalankan live fetch", expanded=False):
        st.caption("Kontrol ini tetap tersedia, tetapi sengaja ditaruh di balik expander agar halaman utama fokus ke insight.")
        col1, col2, col3 = st.columns(3)
        with col1:
            period_mode = st.selectbox(
                "Periode",
                ["7 hari terakhir", "30 hari terakhir", "90 hari terakhir", "Pilih rentang sendiri"],
                index=1,
                key="fetch_period",
            )
            period_map = {
                "7 hari terakhir": "7d",
                "30 hari terakhir": "30d",
                "90 hari terakhir": "90d",
                "Pilih rentang sendiri": "custom",
            }
            today = datetime.today().date()
            chosen_mode = period_map[period_mode]
            auto_start, auto_end = select_period(chosen_mode, today - timedelta(days=30), today)

            if chosen_mode == "custom":
                custom_range = st.date_input(
                    "Rentang tanggal",
                    value=(today - timedelta(days=30), today),
                    key="fetch_range",
                )
                if isinstance(custom_range, tuple) and len(custom_range) == 2:
                    start_custom, end_custom = custom_range
                elif isinstance(custom_range, date):
                    start_custom = custom_range
                    end_custom = custom_range
                else:
                    start_custom, end_custom = today - timedelta(days=30), today
            else:
                start_custom, end_custom = auto_start, auto_end
                st.date_input(
                    "Rentang tanggal (otomatis)",
                    value=(auto_start, auto_end),
                    disabled=True,
                )

        with col2:
            model_id = st.selectbox(
                "Model",
                registry_df["model_id"].tolist(),
                index=int(registry_df["model_id"].tolist().index(default_model_id)),
                format_func=lambda mid: coalesce_text(
                    registry_df.loc[registry_df["model_id"] == mid, "display_name"].iloc[0],
                    mid,
                ),
                key="fetch_model",
            )
            fetch_mode = st.radio(
                "Mode pengambilan",
                ["Representatif (tanpa limit)", "Dengan limit (lebih cepat)"],
                horizontal=True,
                key="fetch_mode",
            )
            use_limit = fetch_mode == "Dengan limit (lebih cepat)"
            fetch_scope = st.radio(
                "Cakupan aplikasi",
                ["Combined", "Kredivo", "Akulaku"],
                horizontal=True,
                key="fetch_scope",
            )

        with col3:
            if use_limit:
                st.info("Mode limit aktif")
            else:
                st.success("Mode representatif aktif: tanpa limit")
            if fetch_scope == "Combined":
                st.caption("Fetch akan menarik dua aplikasi sekaligus: Kredivo dan Akulaku.")
            else:
                st.caption(f"Fetch akan menarik khusus {fetch_scope}.")

        settings_left, settings_right = st.columns([2.2, 1])
        with settings_left:
            if use_limit:
                review_limit_selected = st.select_slider(
                    "Review limit",
                    options=[100, 300, 500, 1000, 2000, 3000, 5000],
                    value=1000,
                )
                review_limit = review_limit_selected
                st.caption(f"Target maksimum total ulasan: {review_limit}")
            else:
                review_limit = None
                st.caption("Mode ini mencoba mengambil ulasan in-range tanpa batas eksplisit, tetap mengikuti filter dan cleaning pipeline.")
        with settings_right:
            allow_cached = st.checkbox("Pakai cache jika scope sama", value=True, key="fetch_use_cache")

        submitted = st.button("Jalankan Fetch + Inference", type="primary", use_container_width=True)

        if submitted:
            model_record = registry_df.loc[registry_df["model_id"] == model_id].iloc[0]
            start_date, end_date = select_period(chosen_mode, start_custom, end_custom)
            if start_date > end_date:
                st.error("Rentang tanggal tidak valid. Tanggal awal harus <= tanggal akhir.")
                return

            if fetch_scope == "Combined":
                app_specs = [("Kredivo", APPS["Kredivo"]), ("Akulaku", APPS["Akulaku"])]
            elif fetch_scope == "Kredivo":
                app_specs = [("Kredivo", APPS["Kredivo"])]
            else:
                app_specs = [("Akulaku", APPS["Akulaku"])]

            status = st.empty()
            progress = st.progress(0)

            def progress_cb(stage: str, scope: str, current: int, total: int) -> None:
                status.info(f"{stage.title()}: {scope} ({current}/{total})")
                if total > 0:
                    progress.progress(min(current / total, 1.0))

            try:
                result = run_live_analysis(
                    store=store,
                    model_id=model_id,
                    app_specs=app_specs,
                    date_from=start_date,
                    date_to=end_date,
                    review_limit=review_limit,
                    predictor_factory=lambda: get_predictor(model_record["source_path"]),
                    progress_cb=progress_cb,
                    allow_cached=allow_cached,
                )
            except Exception as exc:
                st.session_state["fetch_error"] = f"Live run gagal: {exc}"
                st.rerun()

            if result["job_id"]:
                st.session_state["active_job_id"] = result["job_id"]
                st.session_state["scope_job_id"] = result["job_id"]
                st.session_state["pending_scope_job_id"] = result["job_id"]
                if result.get("cached"):
                    st.session_state["fetch_notice"] = f"Scope dimuat dari cache. job_id = {result['job_id']}"
                else:
                    st.session_state["fetch_notice"] = f"Live run selesai. job_id = {result['job_id']}"
                st.rerun()
            else:
                st.session_state["fetch_error"] = "Tidak ada review valid pada periode atau scope ini."
                st.rerun()

    jobs_df = format_jobs(store.list_jobs(), registry_df)
    if jobs_df.empty:
        st.info("Belum ada job live. Gunakan panel fetch di atas untuk membuat scope analisis.")
        return

    with st.container(border=True):
        st.markdown("### Scope Aktif")
        latest_job_id = jobs_df.iloc[0]["job_id"]
        valid_job_ids = jobs_df["job_id"].tolist()
        job_label_map = dict(zip(jobs_df["job_id"], jobs_df["label"]))

        pending_scope_job_id = st.session_state.pop("pending_scope_job_id", None)
        if pending_scope_job_id in valid_job_ids:
            st.session_state["scope_job_id"] = pending_scope_job_id
            st.session_state["active_job_id"] = pending_scope_job_id
        else:
            active_job = st.session_state.get("active_job_id", latest_job_id)
            if active_job not in valid_job_ids:
                active_job = latest_job_id
            if st.session_state.get("scope_job_id") not in valid_job_ids:
                st.session_state["scope_job_id"] = active_job
            st.session_state["active_job_id"] = st.session_state["scope_job_id"]

        scope_source = st.radio(
            "Sumber scope",
            ["Terbaru", "Pilih riwayat"],
            horizontal=True,
            key="scope_source_mode",
            label_visibility="collapsed",
        )

        if scope_source == "Terbaru":
            st.session_state["scope_job_id"] = latest_job_id
            st.session_state["scope_job_picker"] = latest_job_id

        c_scope, c_view = st.columns([2.2, 1])
        with c_scope:
            if scope_source == "Pilih riwayat":
                if st.session_state.get("scope_job_picker") not in valid_job_ids:
                    st.session_state["scope_job_picker"] = st.session_state.get("scope_job_id", latest_job_id)
                st.selectbox(
                    "Scope",
                    valid_job_ids,
                    format_func=lambda job_id: job_label_map.get(job_id, str(job_id)),
                    key="scope_job_picker",
                    label_visibility="collapsed",
                    placeholder="Pilih scope",
                )
                st.session_state["scope_job_id"] = st.session_state["scope_job_picker"]
            else:
                st.text_input(
                    "Scope terbaru",
                    value=job_label_map.get(latest_job_id, latest_job_id),
                    disabled=True,
                    label_visibility="collapsed",
                )
        chosen_job_id = st.session_state["scope_job_id"]
        previous_scope_job_id = st.session_state.get("last_rendered_scope_job_id")
        if previous_scope_job_id != chosen_job_id:
            st.session_state["last_rendered_scope_job_id"] = chosen_job_id
            st.session_state["view_lens_option"] = "Combined"
        st.session_state["active_job_id"] = chosen_job_id
        chosen_job = jobs_df.loc[jobs_df["job_id"] == chosen_job_id].iloc[0]

        reviews_df, preds_df = store.load_job_frames(chosen_job["job_id"])
        long_df_raw = hydrate_scope(reviews_df, preds_df)
        wide_df = wide_review_frame(long_df_raw)
        if long_df_raw.empty or wide_df.empty:
            st.warning("Scope ini belum memiliki data inferensi yang siap ditampilkan.")
            return

        app_counts = (
            wide_df["app_name"].value_counts().rename_axis("app_name").reset_index(name="count")
            if "app_name" in wide_df.columns
            else pd.DataFrame(columns=["app_name", "count"])
        )
        available_apps = app_counts["app_name"].dropna().tolist()
        view_options = ["Combined"] + available_apps if len(available_apps) > 1 else (available_apps or ["Combined"])

        if st.session_state.get("view_lens_option") not in view_options:
            st.session_state["view_lens_option"] = view_options[0]

        with c_view:
            st.selectbox("View", view_options, key="view_lens_option", label_visibility="collapsed")

        if st.session_state["view_lens_option"] != "Combined":
            selected_app = st.session_state["view_lens_option"]
            long_df_raw = long_df_raw[long_df_raw["app_name"] == selected_app].copy()
            wide_df = wide_df[wide_df["app_name"] == selected_app].copy()
            lens_note = f"Menampilkan khusus {selected_app}."
        else:
            if len(available_apps) > 1:
                lens_note = "Menampilkan gabungan semua app dalam scope ini."
            elif len(available_apps) == 1:
                lens_note = f"Scope ini single-app: {available_apps[0]}."
            else:
                lens_note = "Mode tampilan gabungan."

        model_name = registry_df.loc[registry_df["model_id"] == chosen_job["model_id"], "display_name"]
        model_label = coalesce_text(model_name.iloc[0], chosen_job["model_id"]) if not model_name.empty else chosen_job["model_id"]
        st.markdown(
            scope_strip_html(
                chosen_job=chosen_job,
                model_label=str(model_label),
                current_view=str(st.session_state["view_lens_option"]),
                final_reviews=int(wide_df["review_id_ext"].nunique()),
                benchmark_text=benchmark_note(registry_df, chosen_job["model_id"]),
                lens_note=lens_note,
            ),
            unsafe_allow_html=True,
        )
        long_df = filter_present_aspects(long_df_raw)
        presence_df = aspect_presence_summary(long_df_raw)
        if long_df.empty:
            st.warning("Setelah aspect presence filtering, belum ada review yang cukup eksplisit untuk dianalisis per aspek.")
            return
        if not presence_df.empty:
            presence_note = " | ".join(
                f"{aspect_display_name(str(row['aspect']))}: {int(row['present_rows']):,}/{int(row['all_rows']):,} present"
                for _, row in presence_df.iterrows()
            )
            st.caption(
                f"Job aktif: {chosen_job_id}. "
                + "Aspect presence filtering aktif. Hanya review dengan bukti aspek yang cukup dipakai untuk diagnosis per aspek. "
                + presence_note
            )

    kpis = compute_kpis(long_df, wide_df)
    score_df = aspect_score_table(long_df)
    diagnosis_df = aspect_diagnosis_table(long_df, score_df)
    period_start = pd.to_datetime(wide_df["review_date"], errors="coerce").min()
    period_end = pd.to_datetime(wide_df["review_date"], errors="coerce").max()
    duration_days = 0
    if pd.notna(period_start) and pd.notna(period_end):
        duration_days = int((period_end.date() - period_start.date()).days) + 1

    with st.container(border=True):
        st.markdown("### Aspect Health")
        if pd.notna(period_start) and pd.notna(period_end):
            st.caption(
                f"Periode data: {period_start.date()} -> {period_end.date()} "
                f"({duration_days} hari, {kpis['total_reviews']:,} ulasan unik)."
            )
        else:
            st.caption("Ringkasan cepat distribusi sentimen saat ini.")
        health_cards: list[str] = []
        for aspect in ["risk", "trust", "service"]:
            row = score_df[score_df["aspect"] == aspect]
            if row.empty:
                health_cards.append(
                    f'<div class="health-card"><div class="health-name">{ASPECT_LABEL_MAP[aspect]}</div><div class="trust-note">Belum ada data.</div></div>'
                )
                continue
            rec = row.iloc[0].copy()
            rec["aspect"] = ASPECT_LABEL_MAP[aspect]
            health_cards.append(aspect_health_card_html(rec, aspect))
        st.markdown(f'<div class="health-grid">{"".join(health_cards)}</div>', unsafe_allow_html=True)

        summary_html = "".join(
            [
                summary_card_html("Total ulasan unik", f"{kpis['total_reviews']:,}"),
                summary_card_html("Iklim dominan", str(kpis["sentiment_climate"])),
                summary_card_html("Butuh atensi", str(kpis["aspect_pressure"])),
            ]
        )
        st.markdown(f'<div class="summary-grid">{summary_html}</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### Diagnosis Singkat")
        st.caption("Lihat ringkasan per aspek di atas, lalu buka detail jika perlu melihat contoh review.")
        st.markdown('<div class="section-spacer-sm"></div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        aspects = ["risk", "trust", "service"]
        for col, aspect in zip([d1, d2, d3], aspects):
            with col:
                rec = diagnosis_df[diagnosis_df["aspect"] == aspect]
                if rec.empty:
                    st.info(f"Belum ada diagnosis untuk {ASPECT_LABEL_MAP[aspect]}.")
                    continue
                item = rec.iloc[0]
                st.markdown(diagnosis_summary_html(item, aspect), unsafe_allow_html=True)
        st.markdown('<div class="diagnosis-detail-stack"></div>', unsafe_allow_html=True)
        for aspect in aspects:
            rec = diagnosis_df[diagnosis_df["aspect"] == aspect]
            if rec.empty:
                continue
            item = rec.iloc[0]
            with st.expander(f"Detail {item['title']}", expanded=False):
                st.caption(
                    f"Dominan: {item['dominant_sentiment']} ({int(item['dominant_share'])}%) | "
                    f"Volume aspek: {int(item['volume']):,} | "
                    f"Isu utama: {item['issue']}."
                )
                render_diagnosis_examples(long_df, aspect)

    with st.container(border=True):
        st.markdown("### Issue Map per Aspek")
        st.caption("Layer bantu baca review negatif. Ini interpretasi rule-based dengan aspect presence filtering, bukan label gold baru.")
        st.markdown('<div class="section-spacer-sm"></div>', unsafe_allow_html=True)
        i1, i2, i3 = st.columns(3)
        for col, aspect in zip([i1, i2, i3], ["risk", "trust", "service"]):
            full_issue_df = issue_breakdown(long_df, aspect)
            issue_df = full_issue_df.head(3)
            title = ASPECT_LABEL_MAP[aspect]
            with col:
                if issue_df.empty:
                    st.info("Belum ada pola isu.")
                    continue
                st.markdown(
                    issue_map_card_html(title, issue_df, issue_specific_coverage(full_issue_df), aspect),
                    unsafe_allow_html=True,
                )

    with st.container(border=True):
        st.markdown("### Trend Utama")
        st.caption("Satu visual utama untuk memahami apakah komposisi sentimen membaik, stabil, atau memburuk.")
        daily_sent = (
            long_df.groupby([long_df["review_date"].dt.date, "pred_label"], as_index=False)
            .size()
            .rename(columns={"review_date": "date", "size": "count"})
        )
        if not daily_sent.empty:
            daily_sent["date"] = pd.to_datetime(daily_sent["date"])
            daily_sent["share"] = daily_sent.groupby("date")["count"].transform(
                lambda s: (s / s.sum() * 100.0).round(2)
            )
            fig_mix = px.area(
                daily_sent,
                x="date",
                y="share",
                color="pred_label",
                category_orders={"pred_label": ["Negative", "Neutral", "Positive"]},
                color_discrete_map=SENTIMENT_COLOR_MAP,
                title="Komposisi sentimen harian (%)",
            )
            fig_mix.update_yaxes(range=[0, 100], ticksuffix="%")
            st.plotly_chart(chart_theme(fig_mix), use_container_width=True)
        else:
            st.info("Belum ada data trend harian.")

    with st.container(border=True):
        st.markdown("### Summary Kesimpulan")
        st.caption("Ringkasan akhir yang menyatukan pengalaman pakai, area yang perlu dijaga, dan perbedaan antar app.")
        st.markdown('<div class="section-spacer-sm"></div>', unsafe_allow_html=True)
        conclusion_payload = build_summary_conclusion_payload(long_df, score_df)
        st.markdown(str(conclusion_payload["overall"]), unsafe_allow_html=True)
        app_cards = conclusion_payload.get("apps", [])
        if len(app_cards) >= 2:
            st.markdown('<div class="section-spacer-sm"></div>', unsafe_allow_html=True)
            col_left, col_right = st.columns(2, gap="medium")
            with col_left:
                st.markdown(str(app_cards[0]), unsafe_allow_html=True)
            with col_right:
                st.markdown(str(app_cards[1]), unsafe_allow_html=True)
        elif len(app_cards) == 1:
            st.markdown('<div class="section-spacer-sm"></div>', unsafe_allow_html=True)
            st.markdown(str(app_cards[0]), unsafe_allow_html=True)
        signal_html = conclusion_payload.get("signal")
        meaning_html = conclusion_payload.get("meaning")
        if signal_html or meaning_html:
            st.markdown('<div class="section-spacer-sm"></div>', unsafe_allow_html=True)
            if signal_html and meaning_html:
                col_left, col_right = st.columns(2, gap="medium")
                with col_left:
                    st.markdown(str(signal_html), unsafe_allow_html=True)
                with col_right:
                    st.markdown(str(meaning_html), unsafe_allow_html=True)
            elif signal_html:
                st.markdown(str(signal_html), unsafe_allow_html=True)
            elif meaning_html:
                st.markdown(str(meaning_html), unsafe_allow_html=True)

    st.caption("Page 1 sekarang fokus ke distribusi sentimen per aspek, diagnosis yang bisa dibuka-tutup, trend utama, dan kesimpulan ringkas di bagian akhir.")


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError):
        return pd.DataFrame()


def safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def compute_preprocess_funnel() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    report_path = DATA_PROCESSED / "dataset_absa_v2_report.json"
    report = safe_read_json(report_path)

    raw_df = safe_read_csv(DATA_RAW / "reviews_raw.csv")
    clean_v2_df = safe_read_csv(DATA_PROCESSED / "reviews_clean_v2.csv")
    absa_v2_df = safe_read_csv(DATA_PROCESSED / "dataset_absa_v2.csv")
    train_df = safe_read_csv(DATA_PROCESSED / "dataset_absa_50k_v2_intersection.csv")

    raw_stage_name = "Fetched raw reviews"
    if not raw_df.empty:
        fetched = int(len(raw_df))
    else:
        fetched = int(report.get("reviews_clean_v1_rows", 0))
        raw_stage_name = (
            "Raw reviews unavailable (fallback: clean v1 count)"
            if fetched
            else "Fetched raw reviews unavailable"
        )
    clean_v2 = int(len(clean_v2_df)) if not clean_v2_df.empty else int(report.get("reviews_clean_v2_rows", 0))
    labeled_v2 = int(len(absa_v2_df)) if not absa_v2_df.empty else int(report.get("dataset_absa_v2_rows", 0))
    final_train = int(len(train_df))

    aspect_rows = 0
    if not train_df.empty:
        for col in ["risk_sentiment", "trust_sentiment", "service_sentiment"]:
            if col in train_df.columns:
                aspect_rows += int(train_df[col].isin(["Negative", "Neutral", "Positive"]).sum())

    stages = [
        (raw_stage_name, fetched),
        ("After v2 cleaning + dedup", clean_v2),
        ("After v2-label intersection", labeled_v2),
        ("Final train dataset", final_train),
        ("Aspect-level train rows", aspect_rows),
    ]
    funnel_df = pd.DataFrame(stages, columns=["stage", "count"])

    detail_rows = []
    for idx in range(len(stages) - 1):
        before_stage, before_count = stages[idx]
        after_stage, after_count = stages[idx + 1]
        excluded = max(before_count - after_count, 0)
        excl_rate = (excluded / before_count * 100.0) if before_count else 0.0
        detail_rows.append(
            {
                "from_stage": before_stage,
                "to_stage": after_stage,
                "before": before_count,
                "excluded": excluded,
                "after": after_count,
                "exclusion_rate_pct": round(excl_rate, 2),
            }
        )
    detail_df = pd.DataFrame(detail_rows)
    return funnel_df, report, detail_df


def build_live_fetch_audit_views(audit_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    empty_stage_df = pd.DataFrame(columns=["stage_order", "stage_name", "count"])
    empty_detail_df = pd.DataFrame(columns=["from_stage", "to_stage", "before", "excluded", "after", "exclusion_rate_pct"])
    empty_app_df = pd.DataFrame(columns=["app_name", "raw_rows", "final_rows", "removed_total", "removal_rate_pct"])
    if audit_df.empty:
        return empty_stage_df, empty_detail_df, empty_app_df

    stage_df = (
        audit_df.groupby(["stage_order", "stage_name"], as_index=False)["count"]
        .sum()
        .sort_values("stage_order")
        .reset_index(drop=True)
    )

    detail_rows = []
    stage_records = stage_df.to_dict("records")
    for idx in range(len(stage_records) - 1):
        before = int(stage_records[idx]["count"])
        after = int(stage_records[idx + 1]["count"])
        excluded = max(before - after, 0)
        detail_rows.append(
            {
                "from_stage": stage_records[idx]["stage_name"],
                "to_stage": stage_records[idx + 1]["stage_name"],
                "before": before,
                "excluded": excluded,
                "after": after,
                "exclusion_rate_pct": round((excluded / before * 100.0), 2) if before else 0.0,
            }
        )
    detail_df = pd.DataFrame(detail_rows)

    per_app = audit_df.pivot_table(
        index=["app_id", "app_name"],
        columns="stage_order",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    raw_col = 1 if 1 in per_app.columns else None
    final_col = stage_df["stage_order"].max() if not stage_df.empty else None
    if raw_col is not None and final_col is not None and final_col in per_app.columns:
        per_app["raw_rows"] = per_app[raw_col].astype(int)
        per_app["final_rows"] = per_app[final_col].astype(int)
        per_app["removed_total"] = (per_app["raw_rows"] - per_app["final_rows"]).clip(lower=0)
        per_app["removal_rate_pct"] = per_app.apply(
            lambda row: round((row["removed_total"] / row["raw_rows"] * 100.0), 2) if row["raw_rows"] else 0.0,
            axis=1,
        )
        app_df = per_app[["app_name", "raw_rows", "final_rows", "removed_total", "removal_rate_pct"]].copy()
    else:
        app_df = empty_app_df
    return stage_df, detail_df, app_df


def render_preprocess_page(store: DashboardStore, registry_df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Page 2 - Method, Provenance, and Model Trust</div>
            <h2 style="margin:0.2rem 0 0.3rem 0;">Dari mana data datang, diproses bagaimana, dan model mana yang paling valid</h2>
            <div class="section-intro">
                Halaman ini adalah trust layer untuk dashboard: provenance scope, audit live fetch, preprocess funnel, diamond subset, dan perbandingan model gold vs weak-label.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    funnel_df, report, detail_df = compute_preprocess_funnel()
    jobs_df = format_jobs(store.list_jobs(), registry_df)
    gold_summary = get_gold_summary_data()
    gold_subset_df = get_gold_subset_data()
    compare_df = model_compare_frame(registry_df)
    weak_overview_df = get_weak_overview_data()
    gold_aspect_df = gold_per_aspect_frame(gold_summary)
    absent_diag_df = gold_failure_pattern_frame(gold_summary)
    aspect_winner_df = per_aspect_winner_frame(gold_aspect_df, registry_df)

    tab1, tab2, tab3, tab4 = st.tabs(["Fetch & Provenance", "Live Fetch Audit", "Preprocess & Diamond", "Model Compare"])

    with tab1:
        st.markdown("### Scope and Fetch Provenance")
        if jobs_df.empty:
            st.info("Belum ada job live yang tersimpan. Jalankan minimal satu scope di page 1 untuk melihat provenance fetch.")
        else:
            chosen_job_id = st.selectbox(
                "Pilih scope untuk diaudit",
                jobs_df["job_id"].tolist(),
                format_func=lambda job_id: dict(zip(jobs_df["job_id"], jobs_df["label"])).get(job_id, job_id),
                key="research_job_id",
            )
            chosen_job = jobs_df.loc[jobs_df["job_id"] == chosen_job_id].iloc[0]
            reviews_df, preds_df = store.load_job_frames(chosen_job_id)
            long_df = hydrate_scope(reviews_df, preds_df)
            wide_df = wide_review_frame(long_df)
            if {"model_id", "display_name"}.issubset(registry_df.columns):
                display_name = registry_df.loc[registry_df["model_id"] == chosen_job["model_id"], "display_name"]
            else:
                display_name = pd.Series(dtype="object")
            model_label = (
                coalesce_text(display_name.iloc[0], chosen_job["model_id"])
                if not display_name.empty
                else chosen_job["model_id"]
            )

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Job ID", chosen_job["job_id"])
            m2.metric("Scope", chosen_job["app_name"])
            m3.metric("Periode", f"{chosen_job['date_from']} -> {chosen_job['date_to']}")
            m4.metric("Model", model_label)
            m5.metric("Final kept", f"{wide_df['review_id_ext'].nunique():,}" if not wide_df.empty else "0")
            st.caption(
                f"Fetched at {chosen_job['fetched_at']} | review_limit={human_limit_label(chosen_job.get('review_limit'))} | "
                f"{benchmark_note(registry_df, chosen_job['model_id'])}"
            )

            if not wide_df.empty and "app_name" in wide_df.columns:
                app_mix = wide_df["app_name"].value_counts().rename_axis("app_name").reset_index(name="count")
                fig_app_mix = px.bar(app_mix, x="app_name", y="count", title="Komposisi review pada scope aktif")
                fig_app_mix.update_traces(marker_color="#3f6b8a")
                st.plotly_chart(chart_theme(fig_app_mix), use_container_width=True)

        st.markdown("### What Page 1 is Actually Showing")
        st.markdown(
            """
- Scope live fetch yang dipilih user
- Review final yang lolos fetch + cleaning minimum + dedup
- Inferensi ABSA menggunakan model aktif
- Insight non-teknis yang harus tetap bisa ditelusuri balik ke review bukti
- Issue grouping hanyalah layer interpretasi berbasis rule, bukan label gold baru
            """
        )

    with tab2:
        st.markdown("### Live Fetch Audit Trail")
        if jobs_df.empty:
            st.info("Belum ada job live yang tersimpan. Jalankan minimal satu scope di page 1 untuk melihat audit fetch.")
        else:
            audit_job_id = st.selectbox(
                "Pilih live job untuk diaudit",
                jobs_df["job_id"].tolist(),
                format_func=lambda job_id: dict(zip(jobs_df["job_id"], jobs_df["label"])).get(job_id, job_id),
                key="research_audit_job_id",
            )
            audit_job = jobs_df.loc[jobs_df["job_id"] == audit_job_id].iloc[0]
            fetch_audit_df = store.load_live_fetch_audit(audit_job_id)
            stage_df, live_detail_df, live_app_df = build_live_fetch_audit_views(fetch_audit_df)

            if stage_df.empty:
                st.info("Job ini belum punya audit per tahap. Jalankan ulang scope tersebut dengan versi dashboard terbaru untuk mulai menyimpan raw -> filtered trace.")
            else:
                raw_count = int(stage_df.iloc[0]["count"])
                final_count = int(stage_df.iloc[-1]["count"])
                removed_total = max(raw_count - final_count, 0)
                removal_rate = (removed_total / raw_count * 100.0) if raw_count else 0.0
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Job ID", audit_job_id)
                a2.metric("Raw fetched", f"{raw_count:,}")
                a3.metric("Final kept", f"{final_count:,}")
                a4.metric("Total removed", f"{removed_total:,} ({removal_rate:.2f}%)")
                st.caption(
                    f"Scope {audit_job['app_name']} | {audit_job['date_from']} -> {audit_job['date_to']} | "
                    f"model={audit_job['model_id']} | limit={human_limit_label(audit_job.get('review_limit'))}"
                )

                fig_live_funnel = px.funnel(
                    stage_df,
                    x="count",
                    y="stage_name",
                    title="Live fetch volume by stage",
                )
                st.plotly_chart(chart_theme(fig_live_funnel), use_container_width=True)

                st.markdown("### Eliminasi per Tahap")
                st.dataframe(live_detail_df, use_container_width=True, hide_index=True)
                water = live_detail_df.copy()
                water["excluded_negative"] = -water["excluded"]
                fig_live_drop = px.bar(
                    water,
                    x="to_stage",
                    y="excluded_negative",
                    title="Drop per stage pada live fetch (negative = removed)",
                    text_auto=True,
                )
                st.plotly_chart(chart_theme(fig_live_drop), use_container_width=True)

                st.markdown("### Breakdown per App")
                st.dataframe(live_app_df, use_container_width=True, hide_index=True)
                st.markdown(
                    """
1. **API rows fetched**: semua row yang sempat diambil dari Google Play API pada scope itu.
2. **Within date range + non-empty content**: row yang cocok dengan rentang tanggal dan punya isi review.
3. **After raw-text dedup**: duplicate berdasarkan `review_text_raw` dibuang.
4. **After cleaning non-empty**: hasil cleaning yang tidak kosong.
5. **After minimum 3-token filter**: final review yang benar-benar masuk ke inference.
                    """
                )

    with tab3:
        st.markdown("### Preprocess Funnel")
        if funnel_df["count"].sum() == 0:
            st.warning("Data preprocess tidak ditemukan. Pastikan file report/csv preprocess tersedia di data/processed.")
        else:
            headline_a, headline_b, headline_c = st.columns(3)
            headline_a.metric("Fetched raw reviews", f"{int(funnel_df.iloc[0]['count']):,}")
            headline_b.metric("Final train dataset", f"{int(funnel_df.iloc[3]['count']):,}" if len(funnel_df) > 3 else "0")
            headline_c.metric("Aspect-level rows", f"{int(funnel_df.iloc[4]['count']):,}" if len(funnel_df) > 4 else "0")
            fig_funnel = px.funnel(funnel_df, x="count", y="stage", title="Data volume by preprocessing stage")
            st.plotly_chart(chart_theme(fig_funnel), use_container_width=True)

            st.markdown("### Exclusion by Stage")
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
            water = detail_df.copy()
            water["excluded_negative"] = -water["excluded"]
            fig_water = px.bar(
                water,
                x="to_stage",
                y="excluded_negative",
                title="Drop contribution per stage (negative = removed)",
                text_auto=True,
            )
            st.plotly_chart(chart_theme(fig_water), use_container_width=True)

        st.markdown("### Preprocessing Rules and Quality")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Lexicon size", f"{int(report.get('normalization_lexicon_size', 0)):,}")
        q2.metric("Whitelist size", f"{int(report.get('normalization_whitelist_size', 0)):,}")
        q3.metric("Rows with replacement", f"{int(report.get('normalization_rows_with_replacements', 0)):,}")
        q4.metric("Total replacements", f"{int(report.get('normalization_total_replacements', 0)):,}")
        st.markdown(
            """
1. **Fetch raw reviews**: review hasil scrape awal.
2. **Cleaning + dedup**: URL/emoji/newline dibersihkan, junk di-drop, lalu dedup pasca-normalisasi.
3. **Label intersection**: hanya review yang punya silver label valid (`review_id` intersection).
4. **Final train dataset**: subset resmi untuk training.
5. **Aspect-level rows**: satu review bisa menghasilkan lebih dari satu sample karena label per aspek.
            """
        )

        st.markdown("### Diamond Subset Snapshot")
        if gold_subset_df.empty:
            st.info("Diamond subset belum tersedia.")
        else:
            present_df = gold_subset_df[gold_subset_df["aspect_present"] == 1].copy()
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Diamond rows", f"{len(gold_subset_df):,}")
            d2.metric("Aspect present", f"{int((gold_subset_df['aspect_present'] == 1).sum()):,}")
            d3.metric("Aspect absent", f"{int((gold_subset_df['aspect_present'] == 0).sum()):,}")
            d4.metric("Negatif present", f"{int((present_df['label'] == 'Negative').sum()):,}")

            if not present_df.empty:
                aspect_dist = present_df["aspect"].value_counts().rename_axis("aspect").reset_index(name="count")
                label_dist = present_df["label"].value_counts().rename_axis("label").reset_index(name="count")
                p1, p2 = st.columns(2)
                with p1:
                    fig_aspect = px.bar(aspect_dist, x="aspect", y="count", title="Diamond subset per aspect")
                    fig_aspect.update_traces(marker_color="#2f7f52")
                    st.plotly_chart(chart_theme(fig_aspect), use_container_width=True)
                with p2:
                    fig_label = px.bar(label_dist, x="label", y="count", title="Diamond present-only label distribution")
                    fig_label.update_traces(marker_color="#b44e34")
                    st.plotly_chart(chart_theme(fig_label), use_container_width=True)

            st.caption("Catatan penting: source of truth diamond saat ini memakai blank label untuk absent rows; ini lebih kuat daripada guideline lama yang menyebut Neutral.")

    with tab4:
        st.markdown("### Why This Model Is Default")
        default_row = default_model_row(registry_df)
        top_gold = top_model_row_by_metric(compare_df, "gold_f1_macro")
        top_weak = top_model_row_by_metric(compare_df, "weak_f1_macro")
        if default_row is None or compare_df.empty:
            st.info("Belum ada enough registry data untuk menjelaskan pemilihan model default.")
        else:
            default_model_id = str(default_row["model_id"])
            default_model_name = str(default_row["display_name"])
            why1, why2 = st.columns(2)
            with why1:
                st.metric("Model default dashboard", default_model_name)
                st.caption(
                    f"Dipilih karena benchmark utama halaman-1 adalah validitas manusia (gold subset), "
                    f"bukan hanya performa pada weak-label."
                )
            with why2:
                if top_gold is not None and top_weak is not None:
                    st.metric("Konflik benchmark", "Ada" if top_gold["model_id"] != top_weak["model_id"] else "Tidak ada")
                    st.caption(
                        f"Gold winner: {top_gold['display_name']} ({top_gold['gold_f1_macro']:.3f}) | "
                        f"Weak winner: {top_weak['display_name']} ({top_weak['weak_f1_macro']:.3f})"
                    )
                else:
                    st.metric("Konflik benchmark", "Belum tersedia")
                    st.caption("Benchmark gold/weak belum cukup lengkap untuk menentukan winner.")

            st.markdown(
                f"""
- **Keputusan produk**: gunakan **{default_model_name}** untuk page 1 karena dashboard ini harus paling dekat dengan penilaian manusia.
- **Konsekuensi ilmiah**: model terbaik di weak-label tidak otomatis paling tepat untuk insight yang akan dibaca manusia.
- **Makna riset**: uncertainty-aware retraining tetap berguna, tetapi gold evaluation menunjukkan ranking final bisa berubah.
                """
            )

        st.markdown("### Model Comparison")
        if compare_df.empty:
            st.info("Registry model belum tersedia.")
        else:
            c1, c2 = st.columns(2)
            if top_gold is not None:
                c1.metric("Best on gold subset", f"{top_gold['display_name']} ({top_gold['gold_f1_macro']:.3f})")
            else:
                c1.metric("Best on gold subset", "Belum tersedia")
            if top_weak is not None:
                c2.metric("Best on weak-label", f"{top_weak['display_name']} ({top_weak['weak_f1_macro']:.3f})")
            else:
                c2.metric("Best on weak-label", "Belum tersedia")

            st.dataframe(
                compare_df.rename(
                    columns={
                        "display_name": "Model",
                        "gold_f1_macro": "Gold F1 Macro",
                        "gold_accuracy": "Gold Accuracy",
                        "weak_f1_macro": "Weak F1 Macro",
                        "weak_accuracy": "Weak Accuracy",
                        "training_time_seconds": "Training Time (s)",
                        "rank_gold_subset": "Gold Rank",
                        "rank_weak_label": "Weak Rank",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            metric_long = compare_df.melt(
                id_vars=["display_name"],
                value_vars=["gold_f1_macro", "weak_f1_macro"],
                var_name="benchmark",
                value_name="f1_macro",
            )
            metric_long["benchmark"] = metric_long["benchmark"].map(
                {"gold_f1_macro": "Gold subset", "weak_f1_macro": "Weak-label"}
            )
            fig_compare = px.bar(
                metric_long,
                x="display_name",
                y="f1_macro",
                color="benchmark",
                barmode="group",
                title="Macro F1 per model across benchmarks",
            )
            st.plotly_chart(chart_theme(fig_compare), use_container_width=True)

        st.markdown("### Per-Aspect Gold Performance")
        if gold_aspect_df.empty:
            st.info("Ringkasan per-aspect gold belum tersedia.")
        else:
            top_models = compare_df["model_id"].head(4).tolist() if not compare_df.empty else []
            focus_aspect = gold_aspect_df[gold_aspect_df["model_id"].isin(top_models)] if top_models else gold_aspect_df
            fig_aspect = px.bar(
                focus_aspect,
                x="aspect",
                y="f1_macro",
                color="model_id",
                barmode="group",
                title="F1 macro per aspect on gold subset (top models)",
            )
            st.plotly_chart(chart_theme(fig_aspect), use_container_width=True)

        st.markdown("### Per-Aspect Winner")
        if aspect_winner_df.empty:
            st.info("Belum ada per-aspect winner.")
        else:
            st.dataframe(
                aspect_winner_df.rename(
                    columns={
                        "aspect": "Aspect",
                        "model": "Best Model on Gold",
                        "f1_macro": "F1 Macro",
                        "accuracy": "Accuracy",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Known Failure Patterns")
        if absent_diag_df.empty:
            st.info("Belum ada ringkasan absent diagnostics.")
        else:
            st.dataframe(
                absent_diag_df.rename(
                    columns={
                        "model_id": "Model",
                        "absent_rows": "Absent Rows",
                        "absent_mean_confidence": "Mean Confidence on Absent",
                        "pred_negative_when_absent": "Negative Predictions on Absent",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Pattern penting dari evaluasi diamond: model cenderung tetap memberi sentimen kuat pada absent aspect, jadi page 2 harus jujur soal keterbatasan ini."
            )

        st.markdown("### Failure Cases You Should Not Ignore")
        if default_row is None:
            st.info("Model default tidak tersedia.")
        else:
            default_summary = gold_model_summary_lookup(gold_summary, str(default_row["model_id"]))
            case_df = failure_case_frame(default_summary, limit=3)
            if case_df.empty:
                st.info("Belum ada contoh failure case untuk model default.")
            else:
                st.dataframe(case_df, use_container_width=True, hide_index=True)
                st.caption(
                    "Contoh di atas menunjukkan failure mode yang paling berbahaya: model tetap sangat yakin meski aspek yang dinilai sebenarnya tidak hadir secara eksplisit."
                )

        if not weak_overview_df.empty:
            st.caption("Weak-label leaderboard diambil dari epoch comparison summary yang dipakai selama eksperimen training.")


def main() -> None:
    store = get_store()
    registry_df = get_registry()

    with st.sidebar:
        st.markdown("## Fintech Sentiment Observatory")
        page = st.radio("Page", ["Executive ABSA", "Method & Model Trust"], label_visibility="collapsed")
        st.markdown("---")
        st.caption("Halaman-1 fokus ke insight non-teknis dan evidence review. Halaman-2 fokus ke provenance data, preprocess, dan trust model.")

    if page == "Executive ABSA":
        render_all_in_one_page(store, registry_df)
    else:
        render_preprocess_page(store, registry_df)


if __name__ == "__main__":
    main()
