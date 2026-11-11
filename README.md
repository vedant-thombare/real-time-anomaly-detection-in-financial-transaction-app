# Real-Time Financial Transaction Anomaly Detection

A lightweight Python application built in **VS Code** to detect fraudulent or anomalous transactions using unsupervised Machine Learning (**Isolation Forest**) and visualize the results dynamically with **Matplotlib**.

## 📊 Overview

Financial fraud detection requires low latency and high accuracy. This project demonstrates how to identify irregular transactional behavior—such as structural spikes in transaction amounts or transactions occurring at abnormal hours of the day—without needing historical labeled fraud data.

### Key Features
* **Unsupervised Learning:** Uses Scikit-Learn's `IsolationForest` to isolate anomalies based on feature paths.
* **Feature Engineering:** Evaluates transaction risk profiles across two dimensions: `Amount` and `Hour of Day`.
* **Visual Validation:** Flags and renders normal transactions versus high-risk anomalies instantly using Matplotlib charts.

---

## 🛠️ Prerequisites & Setup

Ensure you have **Python 3.8+** installed on your system.

### 1. Environment Setup
Open your terminal in VS Code and run the following commands to initialize a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows (Command Prompt):
venv\Scripts\activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
```

### 2. Dependency Installation
Install the core data science and machine learning packages needed for the project:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## 🚀 How to Run

1. Open **VS Code** and select your project folder.
2. Select your Python interpreter pointing to the virtual environment (`Ctrl + Shift + P` or `Cmd + Shift + P` -> type `Python: Select Interpreter` -> choose the `./venv` option).
3. Execute the script directly from your terminal:

```bash
python app.py
```

---

## 📈 System Architecture & Logic

The system utilizes an **Isolation Forest** architecture which is highly effective for anomaly detection:
1. **Contamination Parameter:** Set at `0.03`, instructing the model that roughly 3% of the inbound data stream contains suspicious outliers.
2. **Scoring:** The model recursively partitions feature spaces. Anomalies require fewer splits to isolate than normal structural clusters, returning a prediction score of `-1` (Anomaly) instead of `1` (Normal).

### Sample Output Visualization
* 🔵 **Blue Dots:** Normal structural transaction thresholds (\$50–\$200 range during standard daylight hours).
* ❌ **Red Crosses:** Anomalies flagged due to unusually extreme amounts (e.g., \$1,000+) or high velocity during off-peak hours (e.g., 3:00 AM).

---

## 📂 Project Structure

```text
financial-anomaly-detection/
├── venv/                 # Virtual environment dependencies
├── app.py                # Main transaction generation and ML pipeline logic
└── README.md             # Project documentation
```
