# -*- coding: utf-8 -*-
"""
SmartBuy Guardian — Streamlit Demo (Competition Edition)

AI 反套路消费决策助手

Run:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
from datetime import datetime, timedelta

from analysis.report_builder import build_report
from ai.advisor import get_advice
from frontend.demo_data import DEMO_CASES

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SmartBuy Guardian",
    page_icon="🛡️",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# Matplotlib 中文配置
# ══════════════════════════════════════════════════════════════════════════════

_FONT_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "PingFang SC",
    "Heiti SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans",
]
for _font in _FONT_CANDIDATES:
    try:
        plt.rcParams["font.sans-serif"] = [_font] + plt.rcParams["font.sans-serif"]
        _fig_test, _ax_test = plt.subplots(figsize=(1, 1))
        _ax_test.set_title("test")
        plt.close(_fig_test)
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# ══════════════════════════════════════════════════════════════════════════════
# CSS injection for card styling
# ══════════════════════════════════════════════════════════════════════════════

def _inject_css():
    st.markdown("""
    <style>
    /* ═══════════════════════════════════════════════════════════════════
       DARK THEME — Global overrides
       ═══════════════════════════════════════════════════════════════════ */
    .stApp {
        background-color: #0B0D12;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Sidebar width ──────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #111318;
        min-width: 240px !important;
        max-width: 240px !important;
    }
    [data-testid="stSidebar"] > div:first-child > div:first-child {
        width: 240px !important;
    }

    /* ═══════════════════════════════════════════════════════════════════
       CARDS — Equal height via flex + dark backgrounds
       ═══════════════════════════════════════════════════════════════════ */

    /* Make the 3-column result row flex so cards fill height equally */
    div[data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        display: flex !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div {
        width: 100%;
    }

    .sb-card {
        background: #161920;
        border: 1px solid #262A33;
        border-radius: 14px;
        padding: 26px 22px 20px 22px;
        margin-bottom: 8px;
        height: 100%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.30);
    }
    .sb-card-red    { border-left: 4px solid #FC8181; }
    .sb-card-yellow { border-left: 4px solid #F6AD55; }
    .sb-card-green  { border-left: 4px solid #68D391; }
    .sb-card-blue   { border-left: 4px solid #63B3ED; }

    .sb-card-title {
        font-size: 1.1em;
        font-weight: 700;
        color: #E2E8F0;
        margin-bottom: 14px;
        letter-spacing: 0.5px;
    }

    /* ═══════════════════════════════════════════════════════════════════
       HERO — Larger, bolder, gradient-text feel
       ═══════════════════════════════════════════════════════════════════ */
    .sb-hero {
        text-align: center;
        padding: 36px 0 12px 0;
    }
    .sb-hero h1 {
        font-size: 2.8em;
        font-weight: 900;
        margin-bottom: 6px;
        color: #FFFFFF;
        letter-spacing: 3px;
        text-shadow: 0 0 40px rgba(99,179,237,0.35);
    }
    .sb-hero .subtitle {
        font-size: 1.25em;
        color: #A0AEC0;
        font-weight: 500;
        letter-spacing: 1px;
    }
    .sb-hero .tagline {
        font-size: 0.95em;
        color: #5A6070;
        margin-top: 6px;
    }

    /* ═══════════════════════════════════════════════════════════════════
       RISK SCORE — big number
       ═══════════════════════════════════════════════════════════════════ */
    .sb-risk-score {
        font-size: 4.0em;
        font-weight: 900;
        text-align: center;
        line-height: 1.05;
        letter-spacing: -1px;
    }
    .sb-risk-score-label {
        font-size: 0.95em;
        text-align: center;
        color: #6B7280;
        margin-top: 2px;
    }

    /* ═══════════════════════════════════════════════════════════════════
       VERDICT BADGE — high-contrast pill
       ═══════════════════════════════════════════════════════════════════ */
    .sb-verdict {
        font-size: 1.25em;
        font-weight: 700;
        text-align: center;
        padding: 12px 24px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .sb-verdict-buy      { background: #1C3D2B; color: #68D391; border: 1px solid #2F5A3C; }
    .sb-verdict-wait     { background: #3D2E0A; color: #F6AD55; border: 1px solid #5A4210; }
    .sb-verdict-dont     { background: #3D1A1A; color: #FC8181; border: 1px solid #5A2A2A; }
    .sb-verdict-unknown  { background: #1A1D24; color: #8899AA; border: 1px solid #2D313A; }

    /* ═══════════════════════════════════════════════════════════════════
       RISK POINTS — warning chips
       ═══════════════════════════════════════════════════════════════════ */
    .sb-risk-point {
        padding: 7px 12px;
        margin: 5px 0;
        background: #2D1A1A;
        border-radius: 8px;
        font-size: 0.9em;
        color: #FC8181;
        border: 1px solid #3D2525;
    }

    /* ═══════════════════════════════════════════════════════════════════
       AI ADVICE — prominent blockquote
       ═══════════════════════════════════════════════════════════════════ */
    .sb-advice-block {
        background: #161920;
        border: 1px solid #262A33;
        border-left: 4px solid #63B3ED;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 1.02em;
        line-height: 1.7;
        color: #CBD5E0;
    }
    .sb-advice-block strong {
        color: #FFFFFF;
    }

    /* Confidence bar */
    .sb-confidence {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 14px;
        padding: 8px 0;
    }
    .sb-confidence-bar {
        flex: 1;
        height: 6px;
        border-radius: 3px;
        background: #262A33;
        overflow: hidden;
    }
    .sb-confidence-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.4s;
    }

    /* ═══════════════════════════════════════════════════════════════════
       OVERRIDE: Streamlit default text & metric colors for dark bg
       ═══════════════════════════════════════════════════════════════════ */
    .stMarkdown, .stCaption, p, span, label, .stRadio label {
        color: #CBD5E0 !important;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }
    [data-testid="stMetricLabel"] {
        color: #6B7280 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #68D391 !important;
    }

    /* Table in dark */
    table {
        color: #CBD5E0 !important;
        background: transparent !important;
    }
    th {
        color: #6B7280 !important;
        font-weight: 600 !important;
    }
    td {
        color: #CBD5E0 !important;
        border-color: #1E2130 !important;
    }
    tr:nth-child(even) td {
        background: rgba(255,255,255,0.02) !important;
    }

    /* Expander in dark */
    [data-testid="stExpander"] {
        background: #161920 !important;
        border: 1px solid #262A33 !important;
        border-radius: 10px !important;
    }
    .streamlit-expanderHeader {
        color: #CBD5E0 !important;
        background: transparent !important;
    }
    .streamlit-expanderContent {
        background: transparent !important;
    }

    /* Radio buttons in dark sidebar */
    .stRadio > div {
        background: transparent !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #CBD5E0 !important;
    }

    /* Horizontal rules */
    hr {
        border-color: #262A33 !important;
    }

    /* Success / Warning / Info / Error callouts */
    [data-testid="stNotification"] {
        background: #161920 !important;
    }
    .stAlert {
        background: #161920 !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0B0D12;
    }
    ::-webkit-scrollbar-thumb {
        background: #2D313A;
        border-radius: 3px;
    }

    /* Code / pre */
    code, pre {
        background: #1A1D24 !important;
        color: #CBD5E0 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: mini price chart (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_mini_chart(prices, current, height=2.0):
    """Render a small inline price history chart (dark theme)."""
    n = len(prices)
    dates = [datetime.now() - timedelta(days=n - i) for i in range(n)]

    BG = "#0B0D12"
    FG = "#CBD5E0"
    GRID = "#1E2230"
    LINE = "#63B3ED"
    FILL = "#63B3ED"
    RED = "#FC8181"
    GREEN = "#68D391"

    fig, ax = plt.subplots(figsize=(7, height))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.fill_between(dates, prices, min(prices) * 0.98,
                    alpha=0.10, color=FILL)
    ax.plot(dates, prices, color=LINE, linewidth=2.0,
            marker="o", markersize=3, markerfacecolor=BG,
            markeredgewidth=1.2, markeredgecolor=LINE)

    ax.scatter([dates[-1]], [current], color=RED, s=70, zorder=10,
               edgecolors=BG, linewidth=2)
    ax.annotate(f"¥{current:.0f}", xy=(dates[-1], current),
                xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=RED, fontweight="bold", va="center")

    min_idx = int(np.argmin(prices))
    ax.scatter([dates[min_idx]], [prices[min_idx]], color=GREEN, s=45, zorder=10)
    ax.annotate(f"最低 ¥{prices[min_idx]:.0f}", xy=(dates[min_idx], prices[min_idx]),
                xytext=(0, -16), textcoords="offset points",
                fontsize=7.5, color=GREEN, ha="center", va="top")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("¥%.0f"))
    ax.tick_params(colors=FG, labelsize=8)
    ax.grid(True, axis="y", alpha=0.25, color=GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)

    y_min, y_max = min(prices), max(prices)
    y_pad = (y_max - y_min) * 0.35
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: run analysis (cached so auto-analysis doesn't re-run on every render)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache(suppress_st_warning=True)
def _run_analysis(product_name, current_price, original_price, history_json, marketing_text):
    """Run analysis engine and return (report, advice). Cached for performance."""
    import json
    history_prices = json.loads(history_json)

    report = build_report(
        product_name=product_name,
        current_price=current_price,
        original_price=original_price if original_price else None,
        history_prices=history_prices,
        marketing_text=marketing_text,
    )
    advice = get_advice(report)
    return report, advice


# ══════════════════════════════════════════════════════════════════════════════
# Helper: render result cards
# ══════════════════════════════════════════════════════════════════════════════

def _verdict_style(verdict_text):
    """Map verdict to CSS class."""
    if "不建议" in verdict_text or "不要" in verdict_text:
        return "sb-verdict-dont"
    if "观望" in verdict_text:
        return "sb-verdict-wait"
    if "建议购买" in verdict_text or "可以" in verdict_text:
        return "sb-verdict-buy"
    return "sb-verdict-unknown"


def _render_result(selected_case, report, advice):
    """Render the 3-card result layout."""
    pa = report["price_analysis"]
    ma = report["marketing_analysis"]
    bd = pa["breakdown"]

    # ── Determine risk colors (dark theme) ─────────────────────────────────
    risk_level = pa["risk_level"]
    if risk_level == "高风险":
        card_class = "sb-card-red"
        risk_color = "#FC8181"
        risk_emoji = "🔴"
    elif risk_level == "中风险":
        card_class = "sb-card-yellow"
        risk_color = "#F6AD55"
        risk_emoji = "🟡"
    else:
        card_class = "sb-card-green"
        risk_color = "#68D391"
        risk_emoji = "🟢"

    # ══════════════════════════════════════════════════════════════════════
    # 3-COLUMN CARD LAYOUT
    # ══════════════════════════════════════════════════════════════════════

    col1, col2, col3 = st.columns([1, 1.1, 1.2])

    # ─── CARD 1: 风险评估 ─────────────────────────────────────────────────
    with col1:
        st.markdown(f"""
        <div class="sb-card {card_class}">
            <div class="sb-card-title">{risk_emoji} 风险评估</div>
            <div class="sb-risk-score" style="color:{risk_color}">{pa['risk_score']:.0f}<span style="font-size:0.35em;color:#5A6070"> /100</span></div>
            <div class="sb-risk-score-label">综合风险评分</div>
            <hr style="margin:12px 0;border-color:#262A33">
            <div class="sb-verdict {_verdict_style(advice.get('verdict',''))}">{advice.get('verdict', '—')}</div>
        </div>
        """, unsafe_allow_html=True)

        # Risk breakdown sub-details
        with st.expander("📋 评分明细"):
            st.markdown(f"""
            | 维度 | 评分 | 权重 |
            |------|------|------|
            | 偏离历史最低 | {bd['deviation_from_low_score']:.0f}/100 | 40% |
            | 偏离历史均价 | {bd['deviation_from_avg_score']:.0f}/100 | 30% |
            | 营销话术风险 | {bd['marketing_risk_score']:.0f}/100 | 30% |
            """)

        # Marketing risk indicators
        mkt_score = ma["risk_score"]
        if mkt_score > 0:
            st.caption(f"📢 营销风险: **{ma['risk_level']}** ({mkt_score:.0f}/100)")

            # Category tags
            cat_tags = []
            for cat_name, cat_data in ma.get("categories", {}).items():
                if cat_data.get("hits"):
                    cat_tags.append(f"`{cat_name}` ×{len(cat_data['hits'])}")
            if cat_tags:
                st.caption("  ".join(cat_tags))

            # Risk highlights (just top 2)
            highlights = ma.get("highlights", [])
            for h in highlights[:2]:
                st.markdown(f'<div class="sb-risk-point">⚠ {h}</div>', unsafe_allow_html=True)

        # Fake original price warning
        if bd.get("fake_original_price"):
            st.warning(f"🚫 {bd['fake_original_detail']}")

    # ─── CARD 2: 价格详情 ─────────────────────────────────────────────────
    with col2:
        st.markdown(f"""
        <div class="sb-card sb-card-blue">
            <div class="sb-card-title">💰 价格详情</div>
        </div>
        """, unsafe_allow_html=True)

        # Key price metrics — use markdown table instead of nested columns
        cur_price = selected_case['current_price']
        lowest = bd['historical_lowest']
        avg_price = bd['historical_average']
        highest = bd['historical_highest']
        dev_low = bd['deviation_from_low_pct']
        trend_label = bd.get('trend_label', '平稳')
        trend_pct = bd.get('trend_pct', 0)

        st.markdown(f"""
        | 指标 | 数值 |
        |------|------|
        | 当前价格 | **¥{cur_price:.0f}** |
        | 历史最低 | ¥{lowest:.0f} |
        | 历史均价 | ¥{avg_price:.0f} |
        | 历史最高 | ¥{highest:.0f} |
        | 偏离历史最低 | {dev_low}% |
        | 近期趋势 | {trend_label} ({trend_pct:+.1f}%) |
        | 记录数 | {bd['record_count']} 条 |
        """)

        # Price chart
        _draw_mini_chart(selected_case["history_prices"], selected_case["current_price"])

        # Original vs actual
        if selected_case.get("original_price"):
            orig = selected_case["original_price"]
            discount = round((orig - cur_price) / orig * 100)
            st.caption(
                f"🏷 标称原价 ¥{orig:.0f} → 折扣约 {discount}% · "
                f"历史最高价 ¥{highest:.0f}"
            )

    # ─── CARD 3: AI 建议 ──────────────────────────────────────────────────
    with col3:
        st.markdown(f"""
        <div class="sb-card sb-card-blue">
            <div class="sb-card-title">🧠 AI 购买建议</div>
        </div>
        """, unsafe_allow_html=True)

        # AI badge
        if advice.get("_source") == "llm":
            st.success("🤖 AI 深度分析")
        else:
            st.info("📐 规则引擎分析")

        # ── Structured advice block ──────────────────────────────────────
        summary = advice.get("summary", "")
        st.markdown(f'<div class="sb-advice-block">{summary}</div>', unsafe_allow_html=True)

        # ── Risk points ──────────────────────────────────────────────────
        risk_points = advice.get("risk_points", [])
        if risk_points:
            st.markdown("**⚠️ 风险提示**")
            for rp in risk_points[:4]:
                st.markdown(f'<div class="sb-risk-point">⚠ {rp}</div>', unsafe_allow_html=True)

        # ── Confidence bar ───────────────────────────────────────────────
        conf = advice.get("confidence", 0)
        conf_pct = int(conf * 100)
        if conf > 0.7:
            conf_color = "#68D391"
        elif conf > 0.4:
            conf_color = "#F6AD55"
        else:
            conf_color = "#FC8181"

        st.markdown(f"""
        <div class="sb-confidence">
            <span style="color:#6B7280;font-size:0.85em">置信度</span>
            <div class="sb-confidence-bar">
                <div class="sb-confidence-fill" style="width:{conf_pct}%;background:{conf_color}"></div>
            </div>
            <span style="color:{conf_color};font-weight:700;font-size:0.95em">{conf:.0%}</span>
        </div>
        """, unsafe_allow_html=True)

        # Source footnote
        if advice.get("_source") != "llm":
            st.caption("💡 配置 LLM API Key 可启用 AI 深度分析")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════

_inject_css()

# ── Session state init ────────────────────────────────────────────────────────
if "selected_case_idx" not in st.session_state:
    st.session_state.selected_case_idx = 0  # default: first case

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📋 Demo 案例")

    # Case selector as radio buttons
    case_names = [c["name"] for c in DEMO_CASES]
    case_labels = [
        f"{'🟢' if c['expectation'].startswith('低风险') else '🔴' if c['expectation'].startswith('高风险') else '🟡'} {c['name']}"
        for c in DEMO_CASES
    ]
    selected_idx = st.radio(
        "选择演示案例",
        range(len(DEMO_CASES)),
        format_func=lambda i: case_labels[i],
        key="case_radio",
    )
    st.session_state.selected_case_idx = selected_idx

    st.markdown("---")

    # Manual input — folded as advanced feature
    with st.expander("🔧 高级：手动输入"):
        st.caption("输入自定义商品信息进行分析")

        manual_name = st.text_input("商品名称", key="manual_name", placeholder="例如：某某蓝牙耳机")
        manual_price = st.number_input("当前价格 (¥)", min_value=0.0, value=0.0, step=1.0, format="%.2f", key="manual_price")
        manual_orig = st.text_input("标称原价 (¥)", key="manual_orig", placeholder="可选")
        manual_mkt = st.text_area("营销文案", key="manual_mkt", placeholder="粘贴促销文字...", height=80)
        manual_hist = st.text_area("历史价格（每行一个）", key="manual_hist", placeholder="399.00\n389.00\n...", height=100)

        if st.button("🔍 分析自定义商品"):
            if not manual_name or manual_price <= 0:
                st.error("请填写商品名称和价格")
            else:
                hist_prices = []
                for line in manual_hist.strip().split("\n"):
                    try:
                        hist_prices.append(float(line.strip()))
                    except ValueError:
                        pass
                if len(hist_prices) < 3:
                    st.error("至少需要 3 条历史价格")
                else:
                    st.session_state.manual_data = {
                        "name": manual_name,
                        "product_name": manual_name,
                        "current_price": manual_price,
                        "original_price": float(manual_orig) if manual_orig.strip() else None,
                        "marketing_text": manual_mkt,
                        "history_prices": hist_prices,
                    }
                    st.session_state.manual_mode = True
                    st.experimental_rerun()

    st.markdown("---")
    st.caption("🛡️ SmartBuy Guardian\n息壤杯 OPC 创新大赛")

# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="sb-hero">
    <h1>🛡️ SmartBuy Guardian</h1>
    <div class="subtitle">AI 反套路消费决策助手</div>
    <div class="tagline">让每一次购物决策都有数据依据</div>
</div>
""", unsafe_allow_html=True)

# Determine which case to show
manual_mode = st.session_state.get("manual_mode", False)
active_case = st.session_state.get("manual_data") if manual_mode else DEMO_CASES[st.session_state.selected_case_idx]

# ══════════════════════════════════════════════════════════════════════════════
# RUN ANALYSIS (auto for cases, triggered for manual)
# ══════════════════════════════════════════════════════════════════════════════

import json

hist_json = json.dumps(active_case["history_prices"])
orig_price = active_case.get("original_price")

with st.spinner("正在分析..."):
    report, advice = _run_analysis(
        active_case["product_name"],
        active_case["current_price"],
        orig_price,
        hist_json,
        active_case["marketing_text"],
    )

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")

# Product name label
product_label = active_case.get("product_name", active_case.get("name", ""))
if manual_mode:
    st.caption(f"📦 当前分析: **{product_label}** · 手动输入模式")
else:
    st.caption(f"📦 当前分析: **{product_label}**")

_render_result(active_case, report, advice)

# Reset manual mode
if manual_mode:
    st.session_state.manual_mode = False

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption(
    "价格诚信检测（规则引擎） + 营销话术分析（关键词库） + AI 购买建议（LLM）"
)
