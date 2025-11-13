"""
ROCm LightGBM Home Credit Loan Repayment Prediction with GPU vs CPU Comparison
"""

import io
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
import gradio as gr
from PIL import Image

plt.switch_backend("Agg")

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
                   num_leaves=2, learning_rate=0.03,
                   num_rounds=1000, max_depth=10,
                   min_data_in_leaf=50, compare_cpu=False):

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

    # Handle missing/categorical
    X = X.fillna(np.nan)
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Drop constant columns
    X = X.loc[:, X.nunique(dropna=False) > 1]

    # Train/validation split
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X, y, applicant_ids, test_size=0.2, random_state=42
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    # Common parameters
    base_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": float(learning_rate),
        "num_leaves": 2,
        "max_depth": int(max_depth),
        "min_data_in_leaf": int(min_data_in_leaf),
        "verbose": -1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "feature_pre_filter": False,
        "enable_bundle": False,
        "use_missing": True,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]

    # -------------------------
    # Train on GPU (ROCm)
    # -------------------------
    gpu_params = base_params.copy()
    gpu_params.update({
        "device_type": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "num_threads": 16,
    })

    print("🚀 Training LightGBM model on ROCm GPU...")
    start_gpu = time.time()
    model_gpu = lgb.train(
        gpu_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=int(num_rounds),
        callbacks=callbacks
    )
    gpu_time = time.time() - start_gpu

    # Evaluate GPU model
    y_pred_proba = model_gpu.predict(X_val, num_iteration=model_gpu.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)
    auc = roc_auc_score(y_val, y_pred_proba)
    acc = accuracy_score(y_val, y_pred)

    # -------------------------
    # Optionally compare with CPU
    # -------------------------
    cpu_time = None
    if compare_cpu:
        cpu_params = base_params.copy()
        cpu_params.update({
            "device_type": "cpu",
            "num_threads": 16,
        })

        print("🧠 Training LightGBM model on CPU for comparison...")
        start_cpu = time.time()
        model_cpu = lgb.train(
            cpu_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=int(num_rounds),
            callbacks=callbacks
        )
        cpu_time = time.time() - start_cpu

    # Repayment probability
    repay_proba = 1 - y_pred_proba
    results_df = pd.DataFrame({
        "SK_ID_CURR": ids_val,
        "Probability_of_Default": y_pred_proba,
        "Probability_of_Repayment": repay_proba,
        "Predicted_Label": np.where(repay_proba >= 0.5, "Likely to Repay", "Risky Applicant")
    }).sort_values(by="Probability_of_Repayment", ascending=True)
    top20 = results_df.head(20).reset_index(drop=True)

    output_csv = "loan_repayment_predictions.csv"
    results_df.to_csv(output_csv, index=False)

    # -------------------------
    # Feature importance plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    lgb.plot_importance(model_gpu, ax=ax, max_num_features=15, importance_type="gain", title="Top 15 Features (GPU)")
    fig.patch.set_facecolor(AMD_BLACK)
    ax.set_facecolor(AMD_BLACK)
    ax.title.set_color(TEXT_WHITE)
    ax.tick_params(colors=TEXT_WHITE)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=AMD_BLACK)
    buf.seek(0)
    feat_img = Image.open(buf)
    plt.close(fig)

    # -------------------------
    # Summary text with timing
    # -------------------------
    summary = (
        f"### 📊 Model Summary\n"
        f"**Train samples:** {len(X_train):,}  **Validation samples:** {len(X_val):,}\n"
        f"**Accuracy:** {acc:.4f}  **AUC:** {auc:.4f}\n"
        f"**🕒 GPU Training Time:** {gpu_time:.2f} seconds\n"
    )
    if cpu_time is not None:
        summary += f"**🧠 CPU Training Time:** {cpu_time:.2f} seconds\n"
        if gpu_time > 0:
            summary += f"**🚀 Speedup (GPU vs CPU):** {cpu_time/gpu_time:.2f}× faster\n"

    summary += f"\n✅ Predictions saved to `{output_csv}`"

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
        <h1 style="margin:0; font-weight: 700;">AMD Instinct MI3xx ROCm-Powered LightGBM Repayment Probability Detection Demo</h1>
        <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/AMD_Logo.svg"
             alt="AMD Logo" style="position: absolute; top: 10px; right: 20px; height: 50px;">
    </div>
    """)

    gr.Markdown("""
    This app:
    1. Loads **application_train.csv** from Home Credit  
    2. Splits into 80% train / 20% validation  
    3. Trains **LightGBM on AMD ROCm GPU**  
    4. Optionally compares with **CPU** performance  
    5. Predicts repayment probability and shows top 20 applicants
    """)

    with gr.Row():
        with gr.Column(scale=2):
            data_path = gr.Textbox(value="application_train.csv", label="Dataset path")
            num_leaves = gr.Number(value=2, label="num_leaves", precision=0)
            learning_rate = gr.Number(value=0.03, label="Learning rate", precision=2)
            num_rounds = gr.Number(value=1000, label="Boost rounds", precision=0)
            max_depth = gr.Number(value=10, label="Max depth", precision=0)
            min_data_in_leaf = gr.Number(value=50, label="Min data in leaf", precision=0)
            compare_cpu = gr.Checkbox(label="Also run on CPU for time comparison", value=False)
            run_btn = gr.Button("🚀 Train and Predict")

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
        inputs=[data_path, num_leaves, learning_rate, num_rounds, max_depth, min_data_in_leaf, compare_cpu],
        outputs=[out_summary, feat_img, auc_val, acc_val, top_table, file_link],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7866, share=False)
