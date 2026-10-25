import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 11

def main():
    print("=== BẮT ĐẦU HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH HỌC MÁY ===")
    os.makedirs('images', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 1. Load data
    data_path = 'alzheimers_disease_data.csv'
    df = pd.read_csv(data_path)
    print(f"Kích thước ban đầu: {df.shape}")

    # Drop PatientID and DoctorInCharge
    data = df.drop(columns=['PatientID', 'DoctorInCharge'])
    print(f"Kích thước sau khi loại bỏ cột không dùng: {data.shape}")

    # Features and Target
    X = data.drop(columns=['Diagnosis'])
    y = data['Diagnosis']

    # 2. Train / Test Split (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Tập huấn luyện X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Tập kiểm thử X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"Phân phối nhãn trước SMOTE trong y_train:\n{y_train.value_counts()}")

    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler and feature names
    joblib.dump(scaler, 'models/scaler.joblib')
    with open('models/features.json', 'w') as f:
        json.dump(list(X.columns), f)

    # 4. SMOTE Oversampling
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    print(f"Kích thước sau SMOTE: X_train_smote: {X_train_smote.shape}, y_train_smote: {y_train_smote.shape}")
    print(f"Phân phối nhãn sau SMOTE:\n{pd.Series(y_train_smote).value_counts()}")

    # 5. Baseline Comparison across Multiple Algorithms
    baseline_models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }

    baseline_results = []
    print("\n--- KẾT QUẢ ĐÁNH GIÁ SƠ BỘ (BASELINE MODELS) ---")
    for name, model in baseline_models.items():
        model.fit(X_train_smote, y_train_smote)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        baseline_results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })
        print(f"{name:25s} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    df_baseline = pd.DataFrame(baseline_results).sort_values(by='F1-Score', ascending=False)
    df_baseline.to_csv('baseline_comparison.csv', index=False)

    # 6. Hyperparameter Tuning with GridSearchCV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- MODEL 1: Random Forest ---
    print("\n>>> Tinh chỉnh siêu tham số RandomForestClassifier...")
    rf_params = {
        'n_estimators': [100, 200, 250],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'criterion': ['gini', 'entropy']
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params,
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )
    rf_grid.fit(X_train_smote, y_train_smote)
    best_rf = rf_grid.best_estimator_
    print(f"Random Forest Best Params: {rf_grid.best_params_}")
    joblib.dump(best_rf, 'models/best_rf.joblib')

    # --- MODEL 2: KNeighborsClassifier ---
    print("\n>>> Tinh chỉnh siêu tham số KNeighborsClassifier...")
    knn_params = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        knn_params,
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )
    knn_grid.fit(X_train_smote, y_train_smote)
    best_knn = knn_grid.best_estimator_
    print(f"KNN Best Params: {knn_grid.best_params_}")
    joblib.dump(best_knn, 'models/best_knn.joblib')

    # --- MODEL 3: SVC ---
    print("\n>>> Tinh chỉnh siêu tham số Support Vector Classification...")
    svc_params = {
        'C': [0.5, 1, 5, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    svc_grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        svc_params,
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )
    svc_grid.fit(X_train_smote, y_train_smote)
    best_svc = svc_grid.best_estimator_
    print(f"SVC Best Params: {svc_grid.best_params_}")
    joblib.dump(best_svc, 'models/best_svc.joblib')

    # 7. Detailed Evaluation on Test Set
    models_to_eval = {
        'RandomForestClassifier': best_rf,
        'KNeighborsClassifier': best_knn,
        'Support Vector Classification': best_svc
    }

    final_metrics = []

    for name, model in models_to_eval.items():
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True)

        final_metrics.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'Best_Params': str(rf_grid.best_params_ if 'RandomForest' in name else (knn_grid.best_params_ if 'KNeighbors' in name else svc_grid.best_params_)),
            'Confusion_Matrix': cm.tolist(),
            'Classification_Report': cr
        })

    with open('models/final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)

    print("\n=== KẾT QUẢ CUỐI CÙNG TRÊN TẬP KIỂM THỬ ===")
    for m in final_metrics:
        print(f"\nModel: {m['Model']}")
        print(f"Best Params: {m['Best_Params']}")
        print(f"Accuracy:  {m['Accuracy']:.4f}")
        print(f"Precision: {m['Precision']:.4f}")
        print(f"Recall:    {m['Recall']:.4f}")
        print(f"F1-Score:  {m['F1-Score']:.4f}")
        print(f"ROC-AUC:   {m['ROC-AUC']:.4f}")
        print("Confusion Matrix:\n", np.array(m['Confusion_Matrix']))

if __name__ == '__main__':
    main()
