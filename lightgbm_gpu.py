import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("application_train.csv")

# Separate target and features
y = df['TARGET']
X = df.drop(columns=['TARGET', 'SK_ID_CURR'])

# Drop constant columns
X = X.loc[:, (X.nunique(dropna=False) > 1)]

# Fill missing values (LightGBM handles NaNs, but safer for GPU training)
X = X.fillna("missing")

# Label encode all object columns (LightGBM on GPU can't handle 'object' dtype)
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Split into train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

# LightGBM GPU parameters for ROCm
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 2,
    'max_bin': 255,
    'max_depth': 10,
    'verbose': -1,
    'device_type': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'num_threads': 16,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,

    # Stability-focused
    'min_data_in_leaf': 20,
    'min_split_gain': 0.0,
    'feature_pre_filter': False,
    'enable_bundle': False,            # Avoids bundling (ROCm buggy here)
    'use_missing': True,              # Forces explicit NaN handling
}


# Callbacks
callbacks = [
    lgb.early_stopping(stopping_rounds=50),
    lgb.log_evaluation(period=100)
]

# Train model
model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=5000,
    callbacks=callbacks
)

# Predict and evaluate

#y_pred_proba[:] = 0.0

# Predictions
y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
y_pred = (y_pred_proba > 0.5).astype(int)

# Metrics
auc = roc_auc_score(y_val, y_pred_proba)
acc = accuracy_score(y_val, y_pred)

print(f"Validation AUC: {auc:.4f}")
print(f"Validation Accuracy: {acc:.4f}")
