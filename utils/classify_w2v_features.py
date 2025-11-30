import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# 2. Классификация Word2Vec
def classify_w2v_features(x_dimensions, y):
    wv_metrics = {'Model': [], 'TP': [], 'FP': [], 'FN': [], 'TN': [],
                  'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'ROC_AUC': []}
    for name, X in zip(['CBOW', 'SG', 'SG_NS'], x_dimensions):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
        lr = LogisticRegression(max_iter=100)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        y_proba = lr.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        wv_metrics['Model'].append(name)
        wv_metrics['TP'].append(tp)
        wv_metrics['FP'].append(fp)
        wv_metrics['FN'].append(fn)
        wv_metrics['TN'].append(tn)
        wv_metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
        wv_metrics['Precision'].append(precision_score(y_test, y_pred))
        wv_metrics['Recall'].append(recall_score(y_test, y_pred))
        wv_metrics['F1'].append(f1_score(y_test, y_pred))
        wv_metrics['ROC_AUC'].append(roc_auc_score(y_test, y_proba))

    return pd.DataFrame(wv_metrics)
