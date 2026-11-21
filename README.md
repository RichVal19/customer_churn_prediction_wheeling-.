# 📉 Customer Churn Risk Predictor

A Streamlit web app that predicts how likely a telecom customer is to **churn** (leave the company).
It serves a **pre-trained Random Forest classifier**: the app only loads the saved model and makes predictions. **No training happens on the platform.**

## Business problem

Winning a new customer costs far more than keeping an existing one. This app helps a retention team spot customers who are likely to leave early enough to act, by turning customer details into a **churn probability** and a **risk level**.

| Risk level | Churn probability |
|------------|-------------------|
| 🟢 LOW     | below 30%         |
| 🟠 MEDIUM  | 30% to 50%        |
| 🔴 HIGH    | 50% or more       |

## Features

- **Single customer prediction:** enter a customer profile and get a churn probability, risk level and suggested action.
- **Batch scoring:** upload a CSV of customers, get a ranked retention list, download the results.
- **Model insight:** top 10 most important features from the Random Forest.

## Model

| | |
|---|---|
| Algorithm | `RandomForestClassifier` (scikit-learn, 100 trees, `random_state=42`) |
| Dataset | [Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) (about 7,000 customers) |
| Target | `Churned` (1 = left, 0 = stayed) |
| Saved file | `Randomforestclassifier.joblib` |
| Training | Done offline in Google Colab (see the project notebook) |

### Input features

| Feature | Type | Values |
|---------|------|--------|
| `tenure` | number | months as a customer |
| `MonthlyCharges` | number | current monthly bill |
| `TotalCharges` | number | total paid so far (the app estimates it as tenure × monthly charges unless you enter it) |
| `SeniorCitizen` | 0 / 1 | 1 = senior |
| `Contract` | category | Month-to-month, One year, Two year |
| `InternetService` | category | DSL, Fiber optic, No |
| `PaymentMethod` | category | Bank transfer (automatic), Credit card (automatic), Electronic check, Mailed check |
| `PaperlessBilling` | category | Yes, No |
| `TechSupport` | category | Yes, No, No internet service |
| `OnlineSecurity` | category | Yes, No, No internet service |

Text columns are **one-hot encoded** in the app exactly as in the notebook (`pd.get_dummies`), and the columns are aligned to the ones the model was trained on.

## Project structure

```
.
├── app.py                          # Streamlit application
├── Randomforestclassifier.joblib   # Pre-trained model (do not retrain)
├── requirements.txt                # Python dependencies
└── README.md
```

## Run locally

```bash
git clone <your-repo-url>
cd <your-repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

The app opens at http://localhost:8501.

## Deploy on Streamlit Community Cloud

1. Push these four files to a GitHub repository (all in the repository root).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose your repository and branch, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit installs the packages from `requirements.txt` and starts the app.

### Important: match the scikit-learn version

A `.joblib` model must be loaded with (nearly) the same scikit-learn version that saved it. In your Colab notebook run:

```python
import sklearn
print(sklearn.__version__)
```

then pin that exact version in `requirements.txt`, for example `scikit-learn==1.6.1`. If you see a loading error or a version warning after deploying, this is the first thing to check.

### Large files on GitHub

Random Forest files can be big. GitHub rejects files above 100 MB; if yours is large, use [Git LFS](https://git-lfs.com) or reduce the file with `joblib.dump(model, "Randomforestclassifier.joblib", compress=3)` in Colab (this does not change the model).

## Batch CSV format

Upload a CSV with one row per customer and these exact column names:

```
tenure, MonthlyCharges, TotalCharges, SeniorCitizen, Contract, InternetService,
PaymentMethod, PaperlessBilling, TechSupport, OnlineSecurity
```

A template can be downloaded from the app. Rows with missing numbers or misspelled categories are skipped and reported.

## Limitations

- The model learns from historical patterns in one dataset; probabilities are estimates, not certainties.
- It only knows the 10 features listed above.
- The default 0.5 decision threshold may miss churners. The risk levels above use a lower 0.3 cut-off for MEDIUM so more at-risk customers are flagged.
- Only load `.joblib` files you created yourself or trust: loading a joblib file can run arbitrary code.

## Tech stack

Python · Streamlit · scikit-learn · pandas · joblib
