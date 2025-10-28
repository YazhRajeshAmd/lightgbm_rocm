"""
ROCm LightGBM Home Credit Loan Repayment Prediction with Gradio UI
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
import gradio as gr
from PIL import Image

plt.switch_backend("Agg")  # headless plotting

# -------------------------
# AMD UI colors
# -------------------------
AMD_TEAL = "#00C2DE"
AMD_BLACK = "#1C1C1C"
TEXT_WHITE = "#FFFFFF"

# -------------------------
# LightGBM training function
# -------------------------
def train_lightgbm(data_path="application_train.csv",
                   num_leaves=15, learning_rate=0.03,
                   num_rounds=1000, max_depth=10, min_data_in_leaf=50):
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        return f"❌ Error loading dataset: {e}", None, None, None, None, None

    if "TARGET" not in df.columns or "SK_ID_CURR" not in df.columns:
        return "❌ Dataset must contain 'TARGET' and 'SK_ID_CURR' columns.", None, None, None, None, None

    # Separate features and target
    y = df["TARGET"]
    X = df.drop(columns=["TARGET"])
    applicant_ids = X["SK_ID_CURR"]
    X = X.drop(columns=["SK_ID_CURR"])

    # Fill missing values as numeric NaN
    X = X.fillna(np.nan)

    # Label encode categorical features
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Drop constant or near-constant columns
    X = X.loc[:, X.nunique(dropna=False) > 1]

    # Train-test split
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X, y, applicant_ids, test_size=0.2, random_state=42
    )

    # LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    # GPU-optimized parameters
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": float(learning_rate),
        "num_leaves": 2,
        "max_depth": int(max_depth),
        "min_data_in_leaf": int(min_data_in_leaf),
        "verbose": -1,
        "device_type": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "num_threads": 16,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "feature_pre_filter": False,
        "enable_bundle": False,
        "use_missing": True,
    }

    # Callbacks
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]

    print("🚀 Training LightGBM model on ROCm GPU...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=5000,
        callbacks=callbacks
    )

    # Predictions
    y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)

    auc = roc_auc_score(y_val, y_pred_proba)
    acc = accuracy_score(y_val, y_pred)

    # Repayment probability
    repay_proba = 1 - y_pred_proba

    results_df = pd.DataFrame({
        "SK_ID_CURR": ids_val,
        "Probability_of_Default": y_pred_proba,
        "Probability_of_Repayment": repay_proba,
        "Predicted_Label": np.where(repay_proba >= 0.5, "Likely to Repay", "Risky Applicant")
    }).sort_values(by="Probability_of_Repayment", ascending=True)

    top20 = results_df.head(20).reset_index(drop=True)

    # Save full predictions to CSV
    output_csv = "loan_repayment_predictions.csv"
    results_df.to_csv(output_csv, index=False)

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(6, 6))
    lgb.plot_importance(model, ax=ax, max_num_features=15, importance_type="gain", title="Top 15 Features")
    fig.patch.set_facecolor(AMD_BLACK)
    ax.set_facecolor(AMD_BLACK)
    ax.title.set_color(TEXT_WHITE)
    ax.tick_params(colors=TEXT_WHITE)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=AMD_BLACK)
    buf.seek(0)
    feat_img = Image.open(buf)
    plt.close(fig)

    summary = (
        f"### 📊 Model Summary\n"
        f"**Train samples:** {len(X_train):,}  **Validation samples:** {len(X_val):,}\n"
        f"**Accuracy:** {acc:.4f}  **AUC:** {auc:.4f}\n\n"
        f"✅ Predictions saved to `{output_csv}`"
    )

    return summary, feat_img, auc, acc, top20, output_csv


# -------------------------
# Gradio UI
# -------------------------
custom_css = f"""
#header {{
    background-color: {AMD_TEAL};
    color: {TEXT_WHITE};
    padding: 10px;
    border-radius: 10px;
    text-align: center;
    font-size: 1.2em;
    font-weight: bold;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
#header img {{
    height: 40px;
}}
footer {{visibility: hidden;}}
.gr-button {{
    background-color: {AMD_TEAL} !important;
    color: {AMD_BLACK} !important;
    font-weight: bold;
}}
"""

with gr.Blocks(css=custom_css, title="ROCm LightGBM Home Credit") as demo:
    gr.HTML(f"""
    <div style="position: relative; padding: 20px; background: linear-gradient(135deg, #f8f9fa 0%, #00c2de 100%); border-radius: 10px; margin-bottom: 20px;">
                <h1 style="margin:0; font-weight: 700;">AMD Instinct MI3xx ROCm-Powered LightGBM Fraud Detection Demo</h1>
                <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/AMD_Logo.svg"
                     alt="AMD Logo" style="position: absolute; top: 10px; right: 20px; height: 50px;">
    </div>
    """)

    gr.Markdown("""
    This app:
    1. Loads **application_train.csv** from Home Credit  
    2. Splits into 80% train / 20% validation  
    3. Trains **LightGBM on ROCm GPU**  
    4. Predicts repayment probability and shows top 20 applicants
    """)

    with gr.Row():
        with gr.Column(scale=2):
            data_path = gr.Textbox(value="application_train.csv", label="Dataset path")
            num_leaves = gr.Number(value=15, label="num_leaves", precision=0)
            learning_rate = gr.Number(value=0.03, label="Learning rate (eta)", precision=2)
            num_rounds = gr.Number(value=1000, label="Boost rounds", precision=0)
            max_depth = gr.Number(value=10, label="Max depth", precision=0)
            min_data_in_leaf = gr.Number(value=50, label="Min data in leaf", precision=0)
            run_btn = gr.Button("🚀 Train and Predict on ROCm GPU")

        with gr.Column(scale=3):
            out_summary = gr.Markdown()
            feat_img = gr.Image(label="Feature Importance")
            auc_val = gr.Number(label="AUC")
            acc_val = gr.Number(label="Accuracy")
            gr.Markdown("### 🔝 Top 20 Applicants by Repayment Probability")
            top_table = gr.Dataframe(label="Top 20 Predictions")
            file_link = gr.File(label="Download All Predictions CSV")

    run_btn.click(
        fn=train_lightgbm,
        inputs=[data_path, num_leaves, learning_rate, num_rounds, max_depth, min_data_in_leaf],
        outputs=[out_summary, feat_img, auc_val, acc_val, top_table, file_link],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7866, share=False)
