import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. Simulate or load financial transaction dataset
np.random.seed(42)
n_samples = 300

# Normal transactions: amount around $50-$200, frequency normal
normal_amounts = np.random.normal(loc=100, scale=25, size=280)
normal_times = np.random.normal(loc=12, scale=3, size=280)

# Anomalous transactions: unusually high/low amounts or odd hours
anomaly_amounts = np.array([600, 750, 550, 5, 2, 850, 900, 480, 510, 620])
anomaly_times = np.array([2, 3, 1, 23, 4, 3, 2, 22, 1, 3])

amounts = np.concatenate([normal_amounts, anomaly_amounts])
hours = np.concatenate([normal_times, anomaly_times])

# Combine into a DataFrame
df = pd.DataFrame({'Amount': amounts, 'Hour': hours})

# 2. Train Isolation Forest for Anomaly Detection
# contamination = expected proportion of outliers in the data
model = IsolationForest(contamination=0.03, random_state=42)
df['Anomaly_Score'] = model.fit_predict(df[['Amount', 'Hour']])

# Scikit-learn outputs -1 for anomalies and 1 for normal transactions
df['Status'] = df['Anomaly_Score'].map({1: 'Normal', -1: 'Anomaly'})

print(df[df['Status'] == 'Anomaly'])

# 3. Visualize results using Matplotlib
plt.figure(figsize=(10, 6))

# Plot normal transactions
normal_data = df[df['Status'] == 'Normal']
plt.scatter(
    normal_data['Hour'],
    normal_data['Amount'],
    color='blue',
    label='Normal Transaction',
    alpha=0.6,
)

# Plot anomalies in red
anomaly_data = df[df['Status'] == 'Anomaly']
plt.scatter(
    anomaly_data['Hour'],
    anomaly_data['Amount'],
    color='red',
    label='Anomaly Detected',
    s=100,
    edgecolor='black',
    marker='X',
)

plt.title('Real-Time Financial Transaction Anomaly Detection', fontsize=14)
plt.xlabel('Hour of Day (0-24)', fontsize=12)
plt.ylabel('Transaction Amount ($)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
