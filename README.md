# 📊 Financial ESG Risk Predictor

A production-ready Streamlit web application that predicts financial and ESG risk levels using pre-trained ML models (Random Forest & Logistic Regression).

---

## 🚀 Features

- **Dual Model Prediction** — Switch between Random Forest and Logistic Regression
- **Live Risk Classification** — High / Medium / Low risk with confidence scores
- **Interactive Visualizations** — Bar charts, radar chart, correlation heatmap
- **Downloadable Reports** — PDF and Excel reports with full analysis
- **Bulk Prediction** — Run predictions on 100-row dataset sample with accuracy metrics
- **No file upload needed** — Dataset is bundled; just tune sliders and predict

---

## 🗂️ Project Structure

```
esg_risk_app/
├── app.py                          # Main Streamlit application
├── financial_esg_risk_dataset.csv  # Training dataset (4,000 records)
├── models/
│   ├── random_forest_model.pkl
│   ├── logistic_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── README.md
```

---

## ⚙️ Input Features

| Feature | Description |
|---|---|
| Revenue Growth | YoY revenue growth rate |
| Debt-to-Equity | Financial leverage ratio |
| Return on Assets | Profitability efficiency |
| Current Ratio | Short-term liquidity |
| Market Volatility | Price fluctuation measure |
| Stock Return | Period stock performance |
| ESG Score | Combined ESG rating (0–1) |

---

## 🏃 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/esg-risk-predictor.git
cd esg-risk-predictor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

App will open at `http://localhost:8501`

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → select your repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** — done! 🎉

> ⚠️ **Important:** Streamlit Cloud requires `scikit-learn==1.6.1` in `requirements.txt` to match the version used when the `.pkl` models were saved. Do not upgrade scikit-learn without retraining and re-saving the models.

---

## 🐳 Deploy on Render / Railway

**Render:**
1. Connect GitHub repo on [render.com](https://render.com)
2. New Web Service → Runtime: Python 3
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

**Railway:**
1. Connect GitHub on [railway.app](https://railway.app)
2. Add variable: `PORT=8501`
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## 📦 Model Details

| Model | Type | Classes |
|---|---|---|
| Random Forest | Ensemble Classifier | High / Medium / Low |
| Logistic Regression | Linear Classifier | High / Medium / Low |

Models were trained on 4,000 records with 7 financial and ESG features, scaled with StandardScaler and labels encoded with LabelEncoder.

---

## 📄 Report Downloads

After prediction, two report formats are available:
- **PDF** — Formatted report with risk badge, confidence scores, input metrics, and interpretation
- **Excel** — Multi-sheet workbook: Summary, Input Metrics, Dataset Overview

---

## 📝 License

MIT License — free to use and modify.
