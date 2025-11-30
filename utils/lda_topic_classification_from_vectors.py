from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

LDA_CLASSIFIERS = [
    'naive_bayes',
    'decision_tree',
    'logistic'
]


def lda_topic_classification_from_vectors(X_vectors, y, num_topics=5, num_epochs=10, name='',
                                          classifier_type='logistic'):
    metrics = {k: [] for k in ["epoch", "TP", "FP", "FN", "TN",
                               "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]}

    # Выбор классификатора
    if classifier_type == 'naive_bayes':
        clf = GaussianNB()
    elif classifier_type == 'decision_tree':
        clf = DecisionTreeClassifier(random_state=42)
    else:
        clf = LogisticRegression(C=1.0, max_iter=100)
        classifier_type = 'logistic'

    for epoch in range(1, num_epochs + 1):
        print(f"Model: {classifier_type}, Epoch: {epoch}")

        # Split the data into training and testing sets
        print("Splitting data into training and test sets...")

        x_train, x_test, y_train, y_test = train_test_split(
            X_vectors, y, test_size=0.5, random_state=epoch, stratify=y
        )

        # Print out the shapes of the train/test sets
        # print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        # print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_pred_prob = clf.predict_proba(x_test)[:, 1]

        # for label, prob in zip(y_pred, y_pred_prob):
        #     print(f"Predicted label: {label}, Predicted probability: {prob}")

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        metrics["epoch"].append(epoch)
        metrics["TP"].append(tp)
        metrics["FP"].append(fp)
        metrics["FN"].append(fn)
        metrics["TN"].append(tn)
        metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["Precision"].append(precision_score(y_test, y_pred, zero_division=0))
        metrics["Recall"].append(recall_score(y_test, y_pred, zero_division=0))
        metrics["F1"].append(f1_score(y_test, y_pred, zero_division=0))
        metrics["ROC_AUC"].append(roc_auc_score(y_test, y_pred_prob))

    # Печать заголовков
    print(f"{'Epoch':<5} {'TP':<3} {'FP':<3} {'FN':<3} {'TN':<3} "
          f"{'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1':<8} {'ROC_AUC':<8}")

    # Печать строк по эпохам
    for i in range(len(metrics["epoch"])):
        print(f"{metrics['epoch'][i]:<5} "
              f"{metrics['TP'][i]:<3} {metrics['FP'][i]:<3} {metrics['FN'][i]:<3} {metrics['TN'][i]:<3} "
              f"{metrics['Accuracy'][i]:<9.4f} "
              f"{metrics['Precision'][i]:<10.4f} "
              f"{metrics['Recall'][i]:<8.4f} "
              f"{metrics['F1'][i]:<8.4f} "
              f"{metrics['ROC_AUC'][i]:<8.4f}")

    return metrics
