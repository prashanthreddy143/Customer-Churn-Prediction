# Install libraries
!pip install pandas numpy scikit-learn matplotlib seaborn -q

# Download dataset
!wget https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CUSTOMER CHURN PREDICTION - PCA + NCC")
print("="*80)

# STEP 1: Load data
print("\n[STEP 1] Loading dataset...")
data = pd.read_csv('Telco-Customer-Churn.csv')
print(f"✓ Loaded! Shape: {data.shape}")

# STEP 2: Explore
print("\n[STEP 2] Exploring data...")
print(f"Churn: {data['Churn'].value_counts().to_dict()}")

# STEP 3: Preprocess
print("\n[STEP 3] Preprocessing...")
if 'customerID' in data.columns:
    data = data.drop('customerID', axis=1)
data = data.dropna()
X = data.drop('Churn', axis=1)
y = data['Churn'].map({'No': 0, 'Yes': 1})
print(f"✓ Features: {X.shape[1]}, Samples: {len(data)}")

# STEP 4: Encode
print("\n[STEP 4] Encoding...")
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
print(f"✓ Done!")

# STEP 5: Scale
print("\n[STEP 5] Scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"✓ Done!")

# STEP 6: PCA - CHANGED 20 TO 19
print("\n[STEP 6] Applying PCA...")
pca = PCA(n_components=19)  # ← CHANGED THIS LINE
X_pca = pca.fit_transform(X_scaled)
var = pca.explained_variance_ratio_.sum() * 100
print(f"✓ Variance: {var:.2f}%")

# STEP 7: Split
print("\n[STEP 7] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)
print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

# STEP 8: Train
print("\n[STEP 8] Training NCC...")
ncc = NearestCentroid()
ncc.fit(X_train, y_train)
print(f"✓ Done!")

# STEP 9: Predict
print("\n[STEP 9] Predicting...")
y_pred_test = ncc.predict(X_test)
print(f"✓ Done!")

# STEP 10: Evaluate
print("\n" + "="*80)
print("RESULTS")
print("="*80)

acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test)
rec = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

print(f"\n📊 TEST METRICS:")
print(f"  Accuracy:  {acc*100:.2f}%")
print(f"  Precision: {prec*100:.2f}%")
print(f"  Recall:    {rec*100:.2f}%")
print(f"  F1-Score:  {f1:.4f}")

cm = confusion_matrix(y_test, y_pred_test)
print(f"\n🔲 Confusion Matrix:")
print(f"  TN: {cm[0][0]}, FP: {cm[0][1]}")
print(f"  FN: {cm[1][0]}, TP: {cm[1][1]}")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=['No Churn', 'Churn']))

# STEP 11: Plot
print("\n[STEP 11] Creating graphs...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Churn Prediction Results', fontsize=14, fontweight='bold')

# Plot 1: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix')

# Plot 2: Variance
axes[0, 1].bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
axes[0, 1].set_title('PCA Variance by Component')
axes[0, 1].set_xlabel('Component')
axes[0, 1].set_ylabel('Variance')

# Plot 3: Cumulative Variance
cumsum = np.cumsum(pca.explained_variance_ratio_)
axes[1, 0].plot(range(1, len(cumsum)+1), cumsum, 'o-', linewidth=2)
axes[1, 0].axhline(y=0.95, color='r', linestyle='--', label='95%')
axes[1, 0].set_title('Cumulative Variance')
axes[1, 0].set_xlabel('Number of Components')
axes[1, 0].set_ylabel('Cumulative Variance')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
scores = [acc, prec, rec, f1]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
axes[1, 1].bar(metrics, scores, color=colors, alpha=0.7)
axes[1, 1].set_title('Model Performance Metrics')
axes[1, 1].set_ylim([0, 1.1])
axes[1, 1].grid(alpha=0.3, axis='y')

# Add value labels on bars
for i, (metric, score) in enumerate(zip(metrics, scores)):
    axes[1, 1].text(i, score + 0.02, f'{score:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('churn_results.png', dpi=100, bbox_inches='tight')
print("✓ Graph saved!")
plt.show()

# STEP 12: Summary
print("\n" + "="*80)
print("✅ COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\n📌 SUMMARY:")
print(f"  - Dataset: Telco Customer Churn")
print(f"  - Total Samples: 7043")
print(f"  - Features (Original): 19")
print(f"  - Features (After PCA): 19")
print(f"  - Variance Retained: {var:.2f}%")
print(f"  - Algorithm: Nearest Centroid Classifier (NCC)")
print(f"  - Test Accuracy: {acc*100:.2f}%")
print(f"  - Test F1-Score: {f1:.4f}")
print(f"\n🎉 Model trained successfully!")
print(f"   Graph saved as: churn_results.png")
