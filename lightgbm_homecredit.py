import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load data
df = pd.read_csv("application_train.csv")

# Separate target and features
y = df['TARGET']
X = df.drop(columns=['TARGET', 'SK_ID_CURR'])

# Identify categorical features
cat_features = [col for col in X.columns if X[col].dtype == 'object']
for col in cat_features:
    X[col] = X[col].astype('category')

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features)

# Parameters
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}

# Callbacks for early stopping
callbacks = [
    lgb.early_stopping(stopping_rounds=50),
    lgb.log_evaluation(period=50)
]

# Train with early stopping via callbacks
model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=1000,
    callbacks=callbacks
)

# Predict and evaluate
y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
auc = roc_auc_score(y_val, y_pred_proba)
print(f"Validation AUC: {auc:.4f}")
