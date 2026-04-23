import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from fpdf import FPDF

warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial ESG Risk Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stMetric { background: white; border-radius: 12px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    .risk-badge-High   { background:#fee2e2; color:#991b1b; padding:6px 18px; border-radius:99px; font-weight:700; font-size:1.1rem; }
    .risk-badge-Medium { background:#fef9c3; color:#854d0e; padding:6px 18px; border-radius:99px; font-weight:700; font-size:1.1rem; }
    .risk-badge-Low    { background:#dcfce7; color:#166534; padding:6px 18px; border-radius:99px; font-weight:700; font-size:1.1rem; }
    .section-header { font-size:1.15rem; font-weight:600; color:#1e3a5f; margin-bottom:8px; }
    .report-box { background:white; border-radius:12px; padding:20px; box-shadow:0 1px 4px rgba(0,0,0,0.08); margin-bottom:16px; }
    div[data-testid="stSidebar"] { background: #1e3a5f; color: white; }
    div[data-testid="stSidebar"] * { color: white !important; }
    div[data-testid="stSidebar"] .stSlider > label { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODELS ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.join(os.path.dirname(__file__), "models")
    with open(os.path.join(base, "random_forest_model.pkl"), "rb") as f:
        rf_model = pickle.load(f)
    with open(os.path.join(base, "logistic_model.pkl"), "rb") as f:
        lr_model = pickle.load(f)
    with open(os.path.join(base, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(base, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    return rf_model, lr_model, scaler, le

@st.cache_data
def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "financial_esg_risk_dataset.csv")
    return pd.read_csv(path)

rf_model, lr_model, scaler, le = load_models()
df_full = load_dataset()

FEATURES = ['revenue_growth', 'debt_to_equity', 'return_on_assets',
            'current_ratio', 'market_volatility', 'stock_return', 'esg_score']

FEATURE_LABELS = {
    'revenue_growth':   'Revenue Growth',
    'debt_to_equity':   'Debt-to-Equity Ratio',
    'return_on_assets': 'Return on Assets (ROA)',
    'current_ratio':    'Current Ratio',
    'market_volatility':'Market Volatility',
    'stock_return':     'Stock Return',
    'esg_score':        'ESG Score',
}

RISK_COLOR = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}

# ─── PREDICTION HELPER ──────────────────────────────────────────────────────────
def predict(values: dict, model_name: str):
    X = np.array([[values[f] for f in FEATURES]])
    X_scaled = scaler.transform(X)
    model = rf_model if model_name == "Random Forest" else lr_model
    pred_enc = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    label = le.inverse_transform([pred_enc])[0]
    proba_dict = {le.inverse_transform([i])[0]: round(float(p) * 100, 1)
                  for i, p in enumerate(proba)}
    return label, proba_dict

# ─── PDF REPORT ─────────────────────────────────────────────────────────────────
def generate_pdf_report(values, risk_label, proba_dict, model_name, company_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header
    pdf.set_fill_color(30, 58, 95)
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_y(10)
    pdf.cell(0, 10, "Financial ESG Risk Assessment Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated on {datetime.now().strftime('%d %B %Y, %H:%M')}", ln=True, align="C")
    pdf.ln(15)

    # Company & Model info
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, f"Company / Portfolio: {company_name}", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Prediction Model: {model_name}", ln=True)
    pdf.ln(4)

    # Risk Level Badge
    colors = {"High": (239, 68, 68), "Medium": (245, 158, 11), "Low": (34, 197, 94)}
    r, g, b = colors[risk_label]
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 12, f"  Predicted Risk Level: {risk_label}  ", ln=True, fill=True, align="C")
    pdf.ln(4)

    # Confidence Scores
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Model Confidence Scores", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for level in ["High", "Medium", "Low"]:
        pct = proba_dict.get(level, 0)
        pdf.cell(60, 7, f"  {level} Risk:", ln=False)
        pdf.cell(0, 7, f"{pct}%", ln=True)
    pdf.ln(4)

    # Input Features
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Input Financial & ESG Metrics", ln=True)
    pdf.set_fill_color(240, 245, 255)
    pdf.set_font("Helvetica", "", 10)
    fill = False
    for feat, label in FEATURE_LABELS.items():
        pdf.set_fill_color(240, 245, 255) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(100, 7, f"  {label}", fill=True, ln=False)
        pdf.cell(0, 7, f"{values[feat]:.4f}", fill=True, ln=True)
        fill = not fill
    pdf.ln(4)

    # Interpretation
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Risk Interpretation", ln=True)
    pdf.set_font("Helvetica", "", 10)
    interp = {
        "High":   "This entity exhibits HIGH risk characteristics. Elevated debt, low ESG scores, and poor financial ratios suggest significant vulnerability. Immediate risk mitigation strategies are recommended.",
        "Medium": "This entity shows MODERATE risk. Some financial or ESG metrics are below optimal thresholds. Monitoring and targeted improvements are advised.",
        "Low":    "This entity demonstrates LOW risk. Strong financial health and ESG performance indicate resilience. Continue maintaining current practices.",
    }
    pdf.multi_cell(0, 6, interp[risk_label])
    pdf.ln(4)

    # Disclaimer
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5,
        "Disclaimer: This report is generated by an AI-powered model for informational purposes only. "
        "It does not constitute financial, legal, or investment advice. Always consult qualified professionals "
        "before making financial decisions.")

    return pdf.output(dest='S').encode('latin-1')


# ─── EXCEL REPORT ───────────────────────────────────────────────────────────────
def generate_excel_report(values, risk_label, proba_dict, model_name, company_name):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Summary sheet
        summary = pd.DataFrame({
            "Field": ["Company/Portfolio", "Model Used", "Predicted Risk Level", "Report Date",
                      "High Risk %", "Medium Risk %", "Low Risk %"],
            "Value": [company_name, model_name, risk_label,
                      datetime.now().strftime('%d-%m-%Y %H:%M'),
                      proba_dict.get("High", 0),
                      proba_dict.get("Medium", 0),
                      proba_dict.get("Low", 0)]
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)

        # Input metrics sheet
        metrics = pd.DataFrame({
            "Metric": [FEATURE_LABELS[f] for f in FEATURES],
            "Value":  [values[f] for f in FEATURES]
        })
        metrics.to_excel(writer, sheet_name="Input Metrics", index=False)

        # Dataset overview sheet
        df_full.describe().round(4).to_excel(writer, sheet_name="Dataset Overview")

    return output.getvalue()


# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    company_name = st.text_input("Company / Portfolio Name", value="Sample Corp")
    model_choice = st.radio("Prediction Model", ["Random Forest", "Logistic Regression"])

    st.markdown("---")
    st.markdown("### 📐 Financial Metrics")

    col_stats = df_full[FEATURES].describe()

    revenue_growth   = st.slider("Revenue Growth",     -0.20, 0.30, 0.05, 0.001, format="%.3f")
    debt_to_equity   = st.slider("Debt-to-Equity",      0.10, 3.00, 1.50, 0.01)
    return_on_assets = st.slider("Return on Assets",   -0.20, 0.30, 0.05, 0.001, format="%.3f")
    current_ratio    = st.slider("Current Ratio",       0.10, 3.00, 1.50, 0.01)
    market_volatility= st.slider("Market Volatility",   0.10, 1.00, 0.50, 0.01)
    stock_return     = st.slider("Stock Return",        -0.20, 0.30, 0.05, 0.001, format="%.3f")
    esg_score        = st.slider("ESG Score",           0.10, 1.00, 0.60, 0.01)

    predict_btn = st.button("🔍 Predict Risk", use_container_width=True, type="primary")

# ─── INPUT VALUES ───────────────────────────────────────────────────────────────
input_values = {
    'revenue_growth':    revenue_growth,
    'debt_to_equity':    debt_to_equity,
    'return_on_assets':  return_on_assets,
    'current_ratio':     current_ratio,
    'market_volatility': market_volatility,
    'stock_return':      stock_return,
    'esg_score':         esg_score,
}

# ─── HEADER ─────────────────────────────────────────────────────────────────────
st.title("📊 Financial ESG Risk Predictor")
st.markdown("Predict financial and ESG-driven risk levels using ML models trained on 4,000 records.")
st.markdown("---")

# ─── DATASET OVERVIEW (always visible) ──────────────────────────────────────────
with st.expander("📁 Dataset Overview", expanded=False):
    tab1, tab2, tab3 = st.tabs(["Distribution", "Correlation Heatmap", "Raw Stats"])
    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        counts = df_full['risk_level'].value_counts()
        axes[0].bar(counts.index, counts.values,
                    color=[RISK_COLOR[r] for r in counts.index], edgecolor='white', linewidth=1.5)
        axes[0].set_title("Risk Level Distribution", fontweight='bold')
        axes[0].set_ylabel("Count")
        for ax in [axes[0]]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                    colors=[RISK_COLOR[r] for r in counts.index],
                    startangle=90, wedgeprops=dict(edgecolor='white', linewidth=1.5))
        axes[1].set_title("Risk Level Share", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(8, 5))
        corr = df_full[FEATURES].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                    linewidths=0.5, cbar_kws={'shrink': 0.8})
        ax.set_title("Feature Correlation Matrix", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.dataframe(df_full[FEATURES].describe().round(4), use_container_width=True)

# ─── PREDICTION SECTION ─────────────────────────────────────────────────────────
if predict_btn or True:   # always show live prediction
    risk_label, proba_dict = predict(input_values, model_choice)

    st.markdown(f"### 🏢 {company_name}")
    st.markdown(f"**Model:** {model_choice}")

    # Risk Badge
    badge_cls = f"risk-badge-{risk_label}"
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:12px; margin:10px 0;'>
        <span style='font-size:1rem; color:#555;'>Predicted Risk Level:</span>
        <span class='{badge_cls}'>{risk_label} Risk</span>
    </div>
    """, unsafe_allow_html=True)

    # Confidence Metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🔴 High Risk",   f"{proba_dict.get('High',0):.1f}%")
    with c2:
        st.metric("🟡 Medium Risk", f"{proba_dict.get('Medium',0):.1f}%")
    with c3:
        st.metric("🟢 Low Risk",    f"{proba_dict.get('Low',0):.1f}%")

    # Charts row
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Confidence Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        levels = ["High", "Medium", "Low"]
        vals   = [proba_dict.get(l, 0) for l in levels]
        bars = ax.bar(levels, vals,
                      color=[RISK_COLOR[l] for l in levels],
                      edgecolor='white', linewidth=1.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.set_ylabel("Probability (%)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#f8fafc')
        fig.patch.set_facecolor('#f8fafc')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("#### Input vs Dataset Average")
        feat_labels = [FEATURE_LABELS[f].split('(')[0].strip() for f in FEATURES]
        user_vals   = [input_values[f] for f in FEATURES]
        mean_vals   = [df_full[f].mean() for f in FEATURES]

        x = np.arange(len(FEATURES))
        width = 0.35
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(x - width/2, user_vals, width, label='Your Input', color='#3b82f6', alpha=0.85)
        ax.bar(x + width/2, mean_vals, width, label='Dataset Avg', color='#94a3b8', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(feat_labels, rotation=30, ha='right', fontsize=7)
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#f8fafc')
        fig.patch.set_facecolor('#f8fafc')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Radar chart
    st.markdown("#### ESG & Risk Radar")
    radar_feats  = FEATURES
    radar_vals   = [input_values[f] for f in radar_feats]
    radar_labels = [FEATURE_LABELS[f].split('(')[0].strip() for f in radar_feats]

    N = len(radar_feats)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    vals_r = radar_vals + [radar_vals[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, vals_r, color=RISK_COLOR[risk_label], linewidth=2)
    ax.fill(angles, vals_r, color=RISK_COLOR[risk_label], alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=8)
    ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('#f8fafc')
    plt.tight_layout()
    col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
    with col_r2:
        st.pyplot(fig)
    plt.close()

    # Interpretation
    st.markdown("---")
    st.markdown("### 💡 Risk Interpretation")
    interp = {
        "High":   "⚠️ **High Risk detected.** This entity has elevated financial and/or ESG risk indicators. Elevated debt-to-equity, poor ROA, low ESG scores, or high market volatility are contributing factors. **Immediate review and risk mitigation is strongly recommended.**",
        "Medium": "🔔 **Moderate Risk detected.** Some metrics fall below optimal levels. Monitor trends closely and consider targeted ESG or financial improvements to reduce risk exposure.",
        "Low":    "✅ **Low Risk detected.** Strong financial fundamentals and ESG scores indicate resilience. Continue maintaining and improving these practices to sustain low-risk status.",
    }
    st.info(interp[risk_label])

    # ─── DOWNLOAD REPORTS ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Download Reports")

    pdf_bytes   = generate_pdf_report(input_values, risk_label, proba_dict, model_choice, company_name)
    excel_bytes = generate_excel_report(input_values, risk_label, proba_dict, model_choice, company_name)

    fname_base = f"{company_name.replace(' ','_')}_ESG_Risk_{datetime.now().strftime('%Y%m%d_%H%M')}"

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{fname_base}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            label="📊 Download Excel Report",
            data=excel_bytes,
            file_name=f"{fname_base}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # ─── BULK PREDICTION ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗂️ Bulk Prediction on Full Dataset Sample")
    with st.expander("Run predictions on dataset sample (100 rows)"):
        sample_df = df_full[FEATURES].sample(100, random_state=42).copy()
        X_sample  = scaler.transform(sample_df)
        model = rf_model if model_choice == "Random Forest" else lr_model
        preds = le.inverse_transform(model.predict(X_sample))
        sample_df['Predicted Risk'] = preds
        sample_df['Actual Risk']    = df_full.loc[sample_df.index, 'risk_level'].values
        sample_df['Match']          = sample_df['Predicted Risk'] == sample_df['Actual Risk']
        acc = sample_df['Match'].mean() * 100
        st.metric("Sample Accuracy", f"{acc:.1f}%")
        st.dataframe(sample_df.round(4), use_container_width=True)

        bulk_excel = io.BytesIO()
        sample_df.round(4).to_excel(bulk_excel, index=True, engine='openpyxl')
        st.download_button(
            "📊 Download Bulk Predictions (Excel)",
            data=bulk_excel.getvalue(),
            file_name=f"bulk_predictions_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#94a3b8; font-size:0.85rem;'>"
    "Financial ESG Risk Predictor · Powered by Random Forest & Logistic Regression · "
    "Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
