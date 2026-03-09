import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc
import matplotlib.pyplot as plt

# 1. Generate Synthetic Cloud Metrics (CPU Usage)
np.random.seed(42)
total_steps = 10000
time_index = pd.date_range("2026-03-01", periods=total_steps, freq="1T") # 1 minute steps

# Base metric: regular daily seasonality + noise
base_cpu = 50 + 20 * np.sin(np.linspace(0, 50 * np.pi, total_steps)) + np.random.normal(0, 5, total_steps)

# Inject anomalies (CPU spikes that lead to incidents)
incident_labels = np.zeros(total_steps)
for _ in range(50): # 50 random incidents
    spike_start = np.random.randint(100, total_steps - 50)
    base_cpu[spike_start:spike_start+10] += np.random.normal(40, 10, 10) # Spike
    incident_labels[spike_start+5:spike_start+15] = 1 # Incident occurs slightly after spike begins

df = pd.DataFrame({'cpu_usage': base_cpu, 'is_incident': incident_labels}, index=time_index)

# 2. Problem Formulation: Sliding Window W and Horizon H
W = 15 # Look back 15 minutes
H = 5  # Predict if incident occurs in the next 5 minutes

# Create target: 1 if any incident in next H steps
df['target'] = df['is_incident'].rolling(window=H, min_periods=1).max().shift(-H)

# Create features based on window W
df['cpu_mean_W'] = df['cpu_usage'].rolling(window=W).mean()
df['cpu_std_W'] = df['cpu_usage'].rolling(window=W).std()
df['cpu_max_W'] = df['cpu_usage'].rolling(window=W).max()
df['cpu_gradient'] = df['cpu_usage'].diff()

# Drop NaN values caused by rolling windows and shifting
df = df.dropna()

# 3. Train/Test Split (Temporal split to avoid data leakage)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

features = ['cpu_usage', 'cpu_mean_W', 'cpu_std_W', 'cpu_max_W', 'cpu_gradient']
X_train, y_train = train_df[features], train_df['target']
X_test, y_test = test_df[features], test_df['target']

# 4. Model Selection and Training
# Random Forest is robust to outliers and captures non linear relationships well
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 5. Evaluation Setup
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))

# Calculate PR-AUC (Crucial for imbalanced alerting systems)
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
print(f"Precision-Recall AUC: {pr_auc:.3f}")
