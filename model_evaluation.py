from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, roc_curve, roc_auc_score


def evaluate_model(trained_model, X_test, y_test, accuracy=True, precision=False, recall=False, f1=False,
                   confusionMatrix=False,
                   classificationReport=False, rocCurve=False, roc_auc=False, all_methods = False):
    """Evaluate the trained model\n
    return dictionary of all evaluation methods\n
    dict["method"]\n

    method:
         accuracy\n
         precision\n
         recall\n
         f1\n
         roc_auc"""
    y_predict = trained_model.predict(X_test)
    accuracy_s = None
    precision_s = None
    recall_s = None
    f1_s = None
    roc_auc_s = None

    if all_methods:
        accuracy = True
        precision = True
        recall = True
        f1 = True
        confusionMatrix = True
        classificationReport = True
        rocCurve = True
        roc_auc = True

    if accuracy:
        accuracy_s = accuracy_score(y_test, y_predict)

    if precision:
        precision_s = precision_score(y_test, y_predict)

    if recall:
        recall_s = recall_score(y_test, y_predict)

    if f1:
        f1_s = f1_score(y_test, y_predict)

    if confusionMatrix:
        print(confusion_matrix(y_test, y_predict))

    if classificationReport:
        print(classification_report(y_test, y_predict))

    if rocCurve:
        print(roc_curve(y_test, y_predict))

    if roc_auc:
        roc_auc_s = roc_auc_score(y_test, y_predict)

    evaluation = {
        "accuracy": accuracy_s,
        "precision": precision_s,
        "recall": recall_s,
        "f1": f1_s,
        "roc_auc": roc_auc_s
    }

    return evaluation
